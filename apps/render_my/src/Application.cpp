
#include "Application.h"

#include <cuda/config.h>
#include <iomanip>

#include "CheckMacros.h"
#include "ConversionArguments.h"
#include "HostKernels.h"
#include "Utils.h"

// STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#ifdef __linux__
#include <dlfcn.h> // RTLD_NOW
#endif

#include <glm/gtc/matrix_access.hpp>

#include "Mesh.h"
#include "Record.h"
#include "cuda/hit_group_data.h"
#include "cuda/light_definition.h"
#include "cuda/vector_math.h"

// CUDA Driver API version of the OpenGL interop header.
#include <MyAssert.h>
#include <cudaGL.h>

static const char* CUDA_PROGRAMS_PATH = "./bin/render_my_core/";

// gltf node's transform
using GltfTransform = std::variant<fastgltf::TRS, fastgltf::math::fmat4x4>;

namespace detail {

// Fill a MaterialData::Texture.
template <typename T>
void parseTextureInfo(const std::vector<cudaTextureObject_t>& samplers, const T& textureInfo,
                      MaterialData::Texture& texture, size_t samplerOffset = 0) {
  size_t texCoordIndex = textureInfo.texCoordIndex;

  // KHR_texture_transform extension data.
  float2 scale = make_float2(1.0f);
  float rotation = 0.0f;
  float2 translation = make_float2(0.0f);

  // Optional KHR_texture_transform extension data.
  if (textureInfo.transform != nullptr) {
    scale.x = textureInfo.transform->uvScale[0];
    scale.y = textureInfo.transform->uvScale[1];

    rotation = textureInfo.transform->rotation;

    translation.x = textureInfo.transform->uvOffset[0];
    translation.y = textureInfo.transform->uvOffset[1];

    // KHR_texture_transform can override the texture coordinate index.
    if (textureInfo.transform->texCoordIndex.has_value()) {
      texCoordIndex = textureInfo.transform->texCoordIndex.value();
    }
  }

  if (NUM_ATTR_TEXCOORDS <= texCoordIndex) {
    std::cerr << "ERROR: detail::parseTextureInfo() Maximum supported texture coordinate index "
                 "exceeded, using 0.\n";
    texCoordIndex = 0; // PERF This means the device code doesn't need to check if the texcoord
                       // index is in the valid range!
  }

  // Map local texture index to global sampler index
  size_t globalSamplerIndex = textureInfo.textureIndex + samplerOffset;
  MY_ASSERT(0 <= globalSamplerIndex && globalSamplerIndex < samplers.size());

  texture.index = static_cast<int>(texCoordIndex);
  // texture.angle       = rotation; // For optional GUI only, needed to recalculate sin and cos
  // below.
  texture.object = samplers[globalSamplerIndex];
  texture.scale = scale;
  texture.rotation = make_float2(sinf(rotation), cosf(rotation));
  texture.translation = translation;
}

/// Convert glTF types to ours.
dev::PrimitiveType toDevPrimitiveType(fastgltf::PrimitiveType t) {
  // TODO rename namespace dev to app (dev sounds like device, confusing)
  switch (t) {
  case fastgltf::PrimitiveType::Points:
    return dev::PrimitiveType::Points;
  case fastgltf::PrimitiveType::Triangles:
    return dev::PrimitiveType::Triangles;
  default:
    return dev::PrimitiveType::Undefined;
  }
}

const std::string& getDevPrimitiveTypeName(fastgltf::PrimitiveType t) {
  static const std::string names[]{
    "Points", "Lines", "LineLoop", "LineStrip", "Triangles", "TriangleStrip", "TriangleFan",
  };
  return names[static_cast<int>(t)];
}

// Build a glm matrix from translation, rotation, scale. PERF: this is slow.
auto makeMatrix = [](const glm::vec3& tr, const glm::quat& rot, const glm::vec3& scale) {
  glm::mat4 mTranslation = glm::translate(glm::mat4(1.0f), tr);
  glm::mat4 mRotation = glm::toMat4(rot);
  glm::mat4 mScale = glm::scale(glm::mat4(1.0f), scale);
  return mTranslation * mRotation * mScale;
};

// Convert gltf transform to GLM.
glm::mat4x4 toGLMTransform(const GltfTransform& transform) {
  glm::mat4x4 mtx;

  // Matrix and TRS values are mutually exclusive according to the spec.
  if (const fastgltf::math::fmat4x4* matrix = std::get_if<fastgltf::math::fmat4x4>(&transform)) {
    mtx = glm::make_mat4x4(matrix->data());
  } else if (const fastgltf::TRS* trs = std::get_if<fastgltf::TRS>(&transform)) {
    // Warning: The quaternion to mat4x4 conversion here is not correct with all versions of GLM.
    // glTF provides the quaternion as (x, y, z, w), which is the same layout GLM used up to version
    // 0.9.9.8. However, with commit 59ddeb7 (May 2021) the default order was changed to (w, x, y,
    // z). You could either define GLM_FORCE_QUAT_DATA_XYZW to return to the old layout, or you
    // could use the recently added static factory constructor glm::quat::wxyz(w, x, y, z), which
    // guarantees the parameter order.
    // =>
    // Using GLM version 0.9.9.9 (or newer) and glm::quat::wxyz(w, x, y, z).
    // If this is not compiling your glm version is too old!
    const auto translation = glm::make_vec3(trs->translation.data());
    const auto rotation =
      glm::quat::wxyz(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]);
    const auto scale = glm::make_vec3(trs->scale.data());
    mtx = makeMatrix(translation, rotation, scale);
  } else {
    std::cerr << "Missing transform " << __FUNCTION__ << std::endl;
    MY_ASSERT(false);
  }
  return mtx;
}
} // namespace detail

void Application::initSheenLUT() {
  // Create the sheen lookup table which is required to weight the sheen sampling.
  m_picSheenLUT = new Picture();

  if (!m_picSheenLUT->load("/data/codes/optix_all/optix_my/OptiX_apps/apps/render_my/data/sheen_lut.hdr",
                           IMAGE_FLAG_2D)) // This frees all images inside an existing Picture.
  {
    delete m_picSheenLUT;
    m_picSheenLUT = nullptr;

    throw std::runtime_error("ERROR: initSheenLUT() Picture::load() failed.");
  }

  // Create a new texture to keep the old texture intact in case anything goes wrong.
  m_texSheenLUT = new Texture(m_allocator);

  m_texSheenLUT->setAddressMode(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
                                CU_TR_ADDRESS_MODE_CLAMP);

  if (!m_texSheenLUT->create(m_picSheenLUT, IMAGE_FLAG_2D | IMAGE_FLAG_SHEEN)) {
    delete m_texSheenLUT;
    m_texSheenLUT = nullptr;

    throw std::runtime_error("ERROR: initSheenLUT Texture::create() failed.");
  }
}

Application::Application(GLFWwindow* window, const Options& options)
    : m_window(window), m_options(options), m_logger(std::cerr) {
  m_pathAsset = options.getFilename();
  m_width = std::max(1, options.getWidthClient());
  m_height = std::max(1, options.getHeightClient());
  m_resolution.x = std::max(1, options.getWidthResolution());
  m_resolution.y = std::max(1, options.getHeightResolution());
  m_isDirtyResolution = true;
  m_launches = options.getLaunches();
  m_interop = options.getInterop();
  m_punctual = options.getPunctual();
  m_missID = options.getMiss();
  m_pathEnv = options.getEnvironment();
  m_showEnvironment = options.getShowEnvironment();
  m_comparisonSamples = options.getComparisonSamples();

  m_iterations.resize(MAX_LAUNCHES);
  m_benchmarkValues.resize(SIZE_BENCHMARK_VALUES);
  
  // Initialize ground size (will be set by createGround())
  m_groundSize = 16.0f; // Default value (matches createGround())

  // Initialize environment light defaults
  m_colorEnv[0] = m_colorEnv[1] = m_colorEnv[2] = 1.0f;
  m_intensityEnv = 1.0f;
  m_rotationEnv[0] = m_rotationEnv[1] = m_rotationEnv[2] = 0.0f;

  m_bufferHost = nullptr; // Allocated inside updateBuffers() when needed.

  // Initialize denoiser variables
  m_denoiser = nullptr;
  m_d_stateDenoiser = 0;
  m_d_scratchDenoiser = 0;
  m_d_denoisedBuffer = 0;
  m_scratchSizeInBytes = 0;
  m_numInputLayers = 1; // At least the beauty buffer is always used
  m_enableDenoiser = true;

  // Tonemapper settings - further adjusted for stronger shadow contrast
  m_gamma = 2.2f;
  m_colorBalance = make_float3(1.0f, 1.0f, 1.0f);
  m_whitePoint = 0.7f; // Further reduced to enhance shadow contrast
  m_burnHighlights = 0.5f; // Further reduced to preserve highlight details
  m_crushBlacks = 0.05f; // Further reduced to preserve shadow details
  m_saturation = 1.0f; // Reduced to prevent oversaturation
  m_brightness = 0.8f; // Further reduced to enhance shadow visibility

  m_guiState = GUI_STATE_NONE;
  m_isVisibleGUI = false; // No GUI toolbar needed

  m_mouseSpeedRatio = 100.0f;
  m_trackball.setSpeedRatio(m_mouseSpeedRatio);

  m_cudaGraphicsResource = nullptr;
  m_sphereRadiusFraction = options.getSphereRadiusFraction();

  // Setup ImGui binding.
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |=
    ImGuiConfigFlags_NavEnableKeyboard; // Use Tab and arrow keys to navigate through widgets.
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  updateFonts();

#ifdef _WIN32
  // HACK Only enable Multi-Viewport under Windows because of
  // https://github.com/ocornut/imgui/wiki/Multi-Viewports#issues
  // "The feature tends to be broken on Linux/X11 with many window managers.
  //  The feature doesn't work in Wayland."
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport/Platform Windows"
#endif
  io.ConfigWindowsResizeFromEdges =
    true; // More consistent window resize behavior, esp. when using multi-viewports.
  io.ConfigWindowsMoveFromTitleBarOnly =
    true; // Prevent moving the GUI window when inadvertently clicking on an empty space.

  // Only initialize ImGui if we have a window (not in headless mode)
  if (window != nullptr) {
    updateFonts();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    // This initializes ImGui resources like the font texture.
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Do nothing.
    ImGui::EndFrame();

    // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable
    // is set.
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      // Platform windows can change the OpenGL context.
      glfwMakeContextCurrent(m_window);
    }
  }

  initCUDA();
  
  // Initialize OpenGL resources (with or without window)
  initOpenGL();
  
  initOptiX();
  initSheenLUT();

  createRandomScene();
  initRenderer();
  
  // Check if asset comparison mode is enabled
  if (options.getAssetComparison()) {
    m_assetComparison = true;
    // Keep environment background visible in comparison mode for proper lighting
    // m_showEnvironment = false; // Don't show HDRI background in comparison mode
    std::cout << "\n=== Asset Comparison Mode Enabled ===" << std::endl;
    std::cout << "HDRI will be used for both lighting and background display" << std::endl;
    std::cout << "Starting asset comparison rendering..." << std::endl;
    renderAssetComparison();
    std::cout << "Asset comparison rendering completed. Exiting..." << std::endl;
    // Exit after completing the comparison
    exit(0);
  }
}

void Application::addAssetsToScene() {
  std::cout << "\n=== Loading GLB Assets ===" << std::endl;
  
  // Record the number of images and samplers from ground material before loading GLB assets
  size_t groundImageCount = m_images.size();
  size_t groundSamplerCount = m_samplers.size();
  
  std::cout << "Images from ground material: " << groundImageCount << std::endl;
  std::cout << "Samplers from ground material: " << groundSamplerCount << std::endl;
  
  // Scan directory for .glb files
  std::vector<std::string> availableGlbFiles;
  try {
    std::filesystem::path glbDir(m_glb_path);
    if (std::filesystem::exists(glbDir) && std::filesystem::is_directory(glbDir)) {
      for (const auto& entry : std::filesystem::directory_iterator(glbDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".glb") {
          availableGlbFiles.push_back(entry.path().string());
        }
      }
    }
  } catch (const std::exception& e) {
    std::cout << "Error scanning GLB directory: " << e.what() << std::endl;
  }
  
  if (availableGlbFiles.empty()) {
    std::cout << "No .glb files found in directory: " << m_glb_path << std::endl;
    return;
  }
  
  std::cout << "Found " << availableGlbFiles.size() << " .glb files in directory" << std::endl;
  
  // Randomly select 2-4 files
  std::random_device rd;
  std::mt19937 gen(rd());
  
  int numAssetsToLoad = std::uniform_int_distribution<int>(2, std::min(4, (int)availableGlbFiles.size()))(gen);
  std::cout << "Randomly selecting " << numAssetsToLoad << " assets to load" << std::endl;
  
  // Shuffle and select random files
  std::shuffle(availableGlbFiles.begin(), availableGlbFiles.end(), gen);
  std::vector<std::string> selectedGlbPaths(availableGlbFiles.begin(), availableGlbFiles.begin() + numAssetsToLoad);
  
  for (size_t i = 0; i < selectedGlbPaths.size(); ++i) {
    std::cout << "Loading GLB " << (i + 1) << ": " << selectedGlbPaths[i] << std::endl;
    try {
      fastgltf::Asset tempAsset = loadGLTFIntoAsset(std::filesystem::path(selectedGlbPaths[i]));
      m_assets.push_back(std::move(tempAsset));
      std::cout << "  ✓ Successfully loaded GLB " << (i + 1) << std::endl;
    } catch (const std::exception& e) {
      std::cout << "  ✗ Failed to load GLB " << (i + 1) << ": " << e.what() << std::endl;
      std::cout << "  Skipping this file and continuing..." << std::endl;
    }
  }
  
  // Check if any assets were successfully loaded
  if (m_assets.empty()) {
    std::cout << "Warning: No GLB assets were successfully loaded!" << std::endl;
    std::cout << "This may be due to unsupported extensions in the GLB files." << std::endl;
    std::cout << "The scene will continue with procedural primitives only." << std::endl;
    return;
  }
  
  std::cout << "Successfully loaded " << m_assets.size() << " GLB assets" << std::endl;
  
  // Initialize images and textures from GLB assets
  initImages();   
  initTextures(groundImageCount, groundSamplerCount);  
  
  // Initialize materials from GLB assets
  initMaterials(groundSamplerCount); 
  
  // NOTE: We skip initMeshes() because we use custom mesh merging logic in createMultipleGLBInstances()
  // initMeshes() would create duplicate meshes and cause index conflicts
  
  // Create combined mesh instances for each GLB asset
  createMultipleGLBInstances();
  
  std::cout << "GLB Assets processing completed successfully" << std::endl;
}

void Application::createMultipleGLBInstances() {
  std::cout << "\n=== Creating GLB Instances ===" << std::endl;
  std::cout << "Current state before GLB processing:" << std::endl;
  std::cout << "  HostMeshes count: " << m_hostMeshes.size() << " (should be 1 for ground)" << std::endl;
  std::cout << "  Materials count: " << m_materials.size() << std::endl;
  std::cout << "  Instances count: " << m_instances.size() << " (should be 1 for ground)" << std::endl;
  
  for (size_t assetIndex = 0; assetIndex < m_assets.size(); ++assetIndex) {
    const fastgltf::Asset& currentAsset = m_assets[assetIndex];
    std::cout << "\nProcessing GLB asset " << assetIndex << " with " << currentAsset.meshes.size()
              << " meshes and " << currentAsset.materials.size() << " materials" << std::endl;
    createGLBInstancesForAsset(assetIndex, const_cast<fastgltf::Asset&>(currentAsset));
  }
  
  std::cout << "\nFinal state after GLB processing:" << std::endl;
  std::cout << "  HostMeshes count: " << m_hostMeshes.size() << std::endl;
  std::cout << "  Materials count: " << m_materials.size() << std::endl;
  std::cout << "  Instances count: " << m_instances.size() << std::endl;
}

void Application::createGLBInstancesForAsset(size_t assetIndex, fastgltf::Asset& asset) {
  std::cout << "Creating GLB instances for asset " << assetIndex
            << " as a single rigid body with preserved materials..." << std::endl;

  // Helper lambda to get local transform of a node
  auto getNodeLocalTransform = [](const fastgltf::Node& node) -> glm::mat4 {
    if (const auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
      return glm::make_mat4x4(matrix->data());
    } else if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
      const auto translation = glm::make_vec3(trs->translation.data());
      const auto rotation = glm::quat::wxyz(trs->rotation[3], trs->rotation[0], 
                                            trs->rotation[1], trs->rotation[2]);
      const auto scale = glm::make_vec3(trs->scale.data());
      return glm::translate(glm::mat4(1.0f), translation) * 
             glm::toMat4(rotation) * 
             glm::scale(glm::mat4(1.0f), scale);
    }
    return glm::mat4(1.0f);
  };

  // Build global transforms for all nodes (considering hierarchy)
  std::vector<glm::mat4> nodeGlobalTransforms(asset.nodes.size(), glm::mat4(1.0f));
  
  // Helper function to recursively compute global transforms
  std::function<void(size_t, const glm::mat4&)> computeNodeTransform = 
    [&](size_t nodeIndex, const glm::mat4& parentTransform) {
      const fastgltf::Node& node = asset.nodes[nodeIndex];
      glm::mat4 localTransform = getNodeLocalTransform(node);
      nodeGlobalTransforms[nodeIndex] = parentTransform * localTransform;
      
      // Process children
      for (size_t childIndex : node.children) {
        computeNodeTransform(childIndex, nodeGlobalTransforms[nodeIndex]);
      }
    };

  // Find root nodes (nodes that are not children of any other node)
  std::vector<bool> isChild(asset.nodes.size(), false);
  for (const auto& node : asset.nodes) {
    for (size_t childIndex : node.children) {
      isChild[childIndex] = true;
    }
  }
  
  // Start traversal from root nodes
  for (size_t i = 0; i < asset.nodes.size(); ++i) {
    if (!isChild[i]) {
      computeNodeTransform(i, glm::mat4(1.0f));
    }
  }

  // Also check scene nodes if they exist
  if (!asset.scenes.empty() && asset.defaultScene.has_value()) {
    const fastgltf::Scene& scene = asset.scenes[asset.defaultScene.value()];
    for (size_t nodeIndex : scene.nodeIndices) {
      computeNodeTransform(nodeIndex, glm::mat4(1.0f));
    }
  }

  // Map meshes to their transforms (a mesh can be instanced by multiple nodes)
  // For simplicity, we'll create separate primitives for each node-mesh pair
  struct MeshInstance {
    size_t meshIndex;
    glm::mat4 transform;
  };
  std::vector<MeshInstance> meshInstances;
  
  for (size_t nodeIndex = 0; nodeIndex < asset.nodes.size(); ++nodeIndex) {
    const fastgltf::Node& node = asset.nodes[nodeIndex];
    if (node.meshIndex.has_value()) {
      meshInstances.push_back({node.meshIndex.value(), nodeGlobalTransforms[nodeIndex]});
    }
  }

  // Create a new combined host mesh for this GLB asset
  dev::HostMesh& combinedMesh = m_hostMeshes.emplace_back();
  combinedMesh.name = "CombinedGLBModel_" + std::to_string(assetIndex);

  // Track bounding box of all transformed vertices
  glm::vec3 minBounds(FLT_MAX);
  glm::vec3 maxBounds(-FLT_MAX);
  
  // Create a single primitive that will contain all mesh parts
  dev::HostPrimitive& combinedPrimitive = combinedMesh.createNewPrimitive(
    dev::PrimitiveType::Triangles, "CombinedPrimitive_" + std::to_string(assetIndex));
  
  // Collect all vertices, indices, and other attributes
  std::vector<glm::vec3> allPositions;
  std::vector<uint32_t> allIndices;
  std::vector<glm::vec3> allNormals;
  std::vector<glm::vec2> allTexcoords;
  uint32_t indexOffset = 0;

  // Process all mesh instances (each node-mesh pair)
  std::cout << "Processing asset " << assetIndex << " with " << meshInstances.size() 
            << " mesh instances" << std::endl;

  // Debug: Print all mesh instances and their transforms
  for (size_t instIdx = 0; instIdx < meshInstances.size(); ++instIdx) {
    const MeshInstance& meshInst = meshInstances[instIdx];
    const fastgltf::Mesh& gltfMesh = asset.meshes[meshInst.meshIndex];
    const glm::mat4& nodeTransform = meshInst.transform;
    
    std::cout << "  Mesh instance " << instIdx << ": mesh " << meshInst.meshIndex 
              << " (" << gltfMesh.name << ")" << std::endl;
    std::cout << "    Transform matrix:" << std::endl;
    std::cout << "      [" << nodeTransform[0][0] << ", " << nodeTransform[0][1] << ", " << nodeTransform[0][2] << ", " << nodeTransform[0][3] << "]" << std::endl;
    std::cout << "      [" << nodeTransform[1][0] << ", " << nodeTransform[1][1] << ", " << nodeTransform[1][2] << ", " << nodeTransform[1][3] << "]" << std::endl;
    std::cout << "      [" << nodeTransform[2][0] << ", " << nodeTransform[2][1] << ", " << nodeTransform[2][2] << ", " << nodeTransform[2][3] << "]" << std::endl;
    std::cout << "      [" << nodeTransform[3][0] << ", " << nodeTransform[3][1] << ", " << nodeTransform[3][2] << ", " << nodeTransform[3][3] << "]" << std::endl;
  }

  // Process all mesh instances and merge them into a single primitive
  for (size_t instIdx = 0; instIdx < meshInstances.size(); ++instIdx) {
    const MeshInstance& meshInst = meshInstances[instIdx];
    const fastgltf::Mesh& gltfMesh = asset.meshes[meshInst.meshIndex];
    const glm::mat4& nodeTransform = meshInst.transform;
    
    std::cout << "  Processing mesh instance " << instIdx << ", mesh " << meshInst.meshIndex 
              << " (" << gltfMesh.name << ")" << std::endl;

    // Process all primitives in this mesh
    for (size_t primIndex = 0; primIndex < gltfMesh.primitives.size(); ++primIndex) {
      const fastgltf::Primitive& primitive = gltfMesh.primitives[primIndex];

      // Get material index and map to global index
      int materialIndex =
        primitive.materialIndex.has_value() ? primitive.materialIndex.value() : -1;
      int globalMaterialIndex = mapToGlobalMaterialIndex(assetIndex, materialIndex);

      std::cout << "    Processing primitive " << primIndex << " with transform applied" << std::endl;
      std::cout << "      Local material index: " << materialIndex << std::endl;
      std::cout << "      Global material index: " << globalMaterialIndex << std::endl;
      std::cout << "      Total materials available: " << m_materials.size() << std::endl;

      // Verify the material exists
      if (globalMaterialIndex >= 0 && globalMaterialIndex < m_materials.size()) {
        std::cout << "      ✓ Material " << globalMaterialIndex << " found in m_materials" << std::endl;
      } else {
        std::cerr << "      ✗ ERROR: Material index " << globalMaterialIndex << " out of bounds!" << std::endl;
      }

      // Create temporary primitive to process attributes
      dev::HostPrimitive tempPrimitive;
      processPrimitiveAttributes(primitive, asset, tempPrimitive, false);

      // Apply node transform and merge into combined arrays
      if (tempPrimitive.positions.h_ptr != nullptr && tempPrimitive.positions.count > 0) {
        glm::vec3* positions = reinterpret_cast<glm::vec3*>(tempPrimitive.positions.h_ptr);
        for (size_t i = 0; i < tempPrimitive.positions.count; ++i) {
          // Transform position by node's global transform
          glm::vec4 transformedPos = nodeTransform * glm::vec4(positions[i], 1.0f);
          allPositions.push_back(glm::vec3(transformedPos));
          
          // Update bounds
          minBounds = glm::min(minBounds, allPositions.back());
          maxBounds = glm::max(maxBounds, allPositions.back());
        }
        
        // Process indices with offset
        if (tempPrimitive.indices.h_ptr != nullptr && tempPrimitive.indices.count > 0) {
          uint32_t* indices = reinterpret_cast<uint32_t*>(tempPrimitive.indices.h_ptr);
          for (size_t i = 0; i < tempPrimitive.indices.count; ++i) {
            allIndices.push_back(indices[i] + indexOffset);
          }
        }
        
        // Process normals if present
        if (tempPrimitive.normals.h_ptr != nullptr && tempPrimitive.normals.count > 0) {
          glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(nodeTransform)));
          glm::vec3* normals = reinterpret_cast<glm::vec3*>(tempPrimitive.normals.h_ptr);
          for (size_t i = 0; i < tempPrimitive.normals.count; ++i) {
            allNormals.push_back(glm::normalize(normalMatrix * normals[i]));
          }
        }
        
        // Process texcoords if present
        if (tempPrimitive.texcoords[0].h_ptr != nullptr && tempPrimitive.texcoords[0].count > 0) {
          glm::vec2* texcoords = reinterpret_cast<glm::vec2*>(tempPrimitive.texcoords[0].h_ptr);
          for (size_t i = 0; i < tempPrimitive.texcoords[0].count; ++i) {
            allTexcoords.push_back(texcoords[i]);
          }
        }
        
        // Update index offset for next mesh part
        indexOffset += tempPrimitive.positions.count;
      }
    }
  }

  // Set material for the combined primitive (use the first material found)
  if (!meshInstances.empty()) {
    const MeshInstance& firstMeshInst = meshInstances[0];
    const fastgltf::Mesh& firstGltfMesh = asset.meshes[firstMeshInst.meshIndex];
    if (!firstGltfMesh.primitives.empty()) {
      const fastgltf::Primitive& firstPrimitive = firstGltfMesh.primitives[0];
      int materialIndex = firstPrimitive.materialIndex.has_value() ? firstPrimitive.materialIndex.value() : -1;
      int globalMaterialIndex = mapToGlobalMaterialIndex(assetIndex, materialIndex);
      combinedPrimitive.indexMaterial = globalMaterialIndex;
      combinedPrimitive.currentMaterial = globalMaterialIndex;
    }
  }

  // Create host buffers for the combined primitive
  if (!allPositions.empty()) {
    // Create positions buffer
    combinedPrimitive.positions.count = allPositions.size();
    combinedPrimitive.positions.size = allPositions.size() * sizeof(glm::vec3);
    combinedPrimitive.positions.h_ptr = new unsigned char[combinedPrimitive.positions.size];
    memcpy(combinedPrimitive.positions.h_ptr, allPositions.data(), combinedPrimitive.positions.size);
    
    // Create indices buffer
    if (!allIndices.empty()) {
      combinedPrimitive.indices.count = allIndices.size();
      combinedPrimitive.indices.size = allIndices.size() * sizeof(uint32_t);
      combinedPrimitive.indices.h_ptr = new unsigned char[combinedPrimitive.indices.size];
      memcpy(combinedPrimitive.indices.h_ptr, allIndices.data(), combinedPrimitive.indices.size);
    }
    
    // Create normals buffer
    if (!allNormals.empty()) {
      combinedPrimitive.normals.count = allNormals.size();
      combinedPrimitive.normals.size = allNormals.size() * sizeof(glm::vec3);
      combinedPrimitive.normals.h_ptr = new unsigned char[combinedPrimitive.normals.size];
      memcpy(combinedPrimitive.normals.h_ptr, allNormals.data(), combinedPrimitive.normals.size);
    }
    
    // Create texcoords buffer
    if (!allTexcoords.empty()) {
      combinedPrimitive.texcoords[0].count = allTexcoords.size();
      combinedPrimitive.texcoords[0].size = allTexcoords.size() * sizeof(glm::vec2);
      combinedPrimitive.texcoords[0].h_ptr = new unsigned char[combinedPrimitive.texcoords[0].size];
      memcpy(combinedPrimitive.texcoords[0].h_ptr, allTexcoords.data(), combinedPrimitive.texcoords[0].size);
    }
  }

  // Calculate model dimensions and center
  glm::vec3 modelSize = maxBounds - minBounds;
  glm::vec3 modelCenter = (minBounds + maxBounds) * 0.5f; // Center of bounding box
  float modelRadius = glm::length(glm::vec2(modelSize.x, modelSize.z)) * 0.5f; // Horizontal radius
  
  std::cout << "  Model size: (" << modelSize.x << ", " << modelSize.y << ", " << modelSize.z << ")"
            << std::endl;
  std::cout << "  Model center: (" << modelCenter.x << ", " << modelCenter.y << ", " << modelCenter.z << ")"
            << std::endl;
  std::cout << "  Model horizontal radius: " << modelRadius << std::endl;

  // Create device mesh for the combined geometry
  dev::KeyTuple key;
  key.idxHostMesh = m_hostMeshes.size() - 1; // Index of the combined mesh
  key.idxSkin = -1;                          // No skin
  key.idxNode = -1;                          // No specific node

  dev::DeviceMesh& deviceMesh = m_deviceMeshes.emplace_back();
  createDeviceMesh(deviceMesh, key);
  m_mapKeyTupleToDeviceMeshIndex[key] = m_deviceMeshes.size() - 1;

  // Build GAS for the combined device mesh
  buildDeviceMeshAccel(m_deviceMeshes.size() - 1, true);

  // Create an instance for this GLB asset
  dev::Instance instance;

  // Get circle parameters
  float circleRadius = 1.0f; // Default
  glm::vec3 circleCenter(0.0f);
  
  if (assetIndex < m_randomCircles.size()) {
    circleRadius = m_randomCircles[assetIndex].radius;
    circleCenter = m_randomCircles[assetIndex].center;
  } else {
    // Fallback: place at origin with offset
    circleCenter = glm::vec3(assetIndex * 2.0f, 0.0f, 0.0f);
  }

  // Calculate scale to fit model inside circle (with some margin)
  const float margin = 0.95f; // Use 95% of circle radius to leave some space
  float targetRadius = circleRadius * margin;
  float horizontalScaleFactor = (modelRadius > 0.0f) ? (targetRadius / modelRadius) : 1.0f;
  
  // Limit height to prevent camera occlusion
  const float maxHeight = 8.0f; // Maximum height to prevent camera occlusion
  float modelHeight = modelSize.y;
  float heightScaleFactor = (modelHeight > 0.0f) ? (maxHeight / modelHeight) : 1.0f;
  
  // Use the smaller scale factor to ensure both horizontal and vertical constraints are met
  float scaleFactor = std::min(horizontalScaleFactor, heightScaleFactor);
  
  std::cout << "  Circle radius: " << circleRadius << ", target radius: " << targetRadius << std::endl;
  std::cout << "  Model height: " << modelHeight << ", max height: " << maxHeight << std::endl;
  std::cout << "  Horizontal scale factor: " << horizontalScaleFactor << ", height scale factor: " << heightScaleFactor << std::endl;
  std::cout << "  Final scale factor: " << scaleFactor << " (constrained by " << (horizontalScaleFactor < heightScaleFactor ? "horizontal" : "height") << ")" << std::endl;

  // Calculate position: 
  // 1. Model center (in model space) should be at circle center (in world space, XZ plane)
  // 2. Model bottom should be on ground (Y=0)
  
  // After scaling, the model will be transformed:
  //   - XZ: model center should align with circle center
  //   - Y: model bottom should be at ground level (Y=0)
  float scaledMinY = minBounds.y * scaleFactor;
  
  // Final position: circle center in XZ, adjust Y so bottom is at ground
  glm::vec3 finalPosition(
    circleCenter.x - modelCenter.x * scaleFactor,  // Center in X
    -scaledMinY,                                     // Bottom at Y=0
    circleCenter.z - modelCenter.z * scaleFactor   // Center in Z
  );

  // Create transform matrix: CORRECT ORDER - scale first (at origin), then translate
  // Note: GLM matrix multiplication order: first scale, then translate means translate * scale
  glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor));
  glm::mat4 translateMatrix = glm::translate(glm::mat4(1.0f), finalPosition);
  glm::mat4 transform = translateMatrix * scaleMatrix; // Apply scale first, then translate

  instance.transform = transform;
  instance.indexDeviceMesh = m_deviceMeshes.size() - 1;

  m_instances.push_back(instance);

  std::cout << "Created instance for GLB asset " << assetIndex << " at position ("
            << finalPosition.x << ", " << finalPosition.y << ", " << finalPosition.z << ")"
            << std::endl;
  std::cout << "  Model bounds: min(" << minBounds.x << ", " << minBounds.y << ", " << minBounds.z
            << "), max(" << maxBounds.x << ", " << maxBounds.y << ", " << maxBounds.z << ")"
            << std::endl;
}

int Application::mapToGlobalMaterialIndex(size_t assetIndex, int localMaterialIndex) {
  if (localMaterialIndex < 0)
    return -1;

  // Start with offset of 1 to account for the ground material at index 0
  int materialOffset = 1; // Ground material is always at index 0
  
  // Add materials from all previous assets
  for (size_t i = 0; i < assetIndex; ++i) {
    materialOffset += m_assets[i].materials.size();
  }
  
  return materialOffset + localMaterialIndex;
}

void Application::processPrimitiveAttributes(const fastgltf::Primitive& primitive,
                                             fastgltf::Asset& asset, dev::HostPrimitive& hostPrim,
                                             bool fullMode) {
  // Get position data (required)
  auto itPosition = primitive.findAttribute("POSITION");
  if (itPosition != primitive.attributes.end()) {
    int indexAccessor = static_cast<int>(itPosition->accessorIndex);
    utils::createHostBuffer("POSITION", asset, indexAccessor, fastgltf::AccessorType::Vec3,
                            fastgltf::ComponentType::Float, 1.0f, hostPrim.positions);
  }

  // Get index data
  if (primitive.indicesAccessor.has_value()) {
    int indexAccessor = static_cast<int>(primitive.indicesAccessor.value());
    utils::createHostBuffer("INDICES", asset, indexAccessor, fastgltf::AccessorType::Scalar,
                            fastgltf::ComponentType::UnsignedInt, 0.0f, hostPrim.indices);
  }

  // Get normal data
  auto itNormal = primitive.findAttribute("NORMAL");
  if (itNormal != primitive.attributes.end()) {
    int indexAccessor = static_cast<int>(itNormal->accessorIndex);
    utils::createHostBuffer("NORMAL", asset, indexAccessor, fastgltf::AccessorType::Vec3,
                            fastgltf::ComponentType::Float, 0.0f, hostPrim.normals);
  }

  // Get tangent data
  auto itTangent = primitive.findAttribute("TANGENT");
  if (itTangent != primitive.attributes.end()) {
    int indexAccessor = static_cast<int>(itTangent->accessorIndex);
    utils::createHostBuffer("TANGENT", asset, indexAccessor, fastgltf::AccessorType::Vec4,
                            fastgltf::ComponentType::Float, 1.0f, hostPrim.tangents);
  }

  // Get color data
  auto itColor = primitive.findAttribute("COLOR_0");
  if (itColor != primitive.attributes.end()) {
    int indexAccessor = static_cast<int>(itColor->accessorIndex);
    utils::createHostBuffer("COLOR_0", asset, indexAccessor, fastgltf::AccessorType::Vec4,
                            fastgltf::ComponentType::Float, 1.0f, hostPrim.colors);
  }

  // Get texture coordinate data
  for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j) {
    std::string strTexcoord = "TEXCOORD_" + std::to_string(j);
    auto itTexcoord = primitive.findAttribute(strTexcoord);
    if (itTexcoord != primitive.attributes.end()) {
      int indexAccessor = static_cast<int>(itTexcoord->accessorIndex);
      utils::createHostBuffer(strTexcoord.c_str(), asset, indexAccessor,
                              fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float, 0.0f,
                              hostPrim.texcoords[j]);
    }
  }

  // Full mode includes additional features like morphing and skinning
  if (fullMode) {
    // Process joints and weights for skinning
    for (int j = 0; j < NUM_ATTR_JOINTS; ++j) {
      std::string joints_str = "JOINTS_" + std::to_string(j);
      auto itJoints = primitive.findAttribute(joints_str);
      if (itJoints != primitive.attributes.end()) {
        int indexAccessor = static_cast<int>(itJoints->accessorIndex);
        utils::createHostBuffer(joints_str.c_str(), asset, indexAccessor,
                                fastgltf::AccessorType::Vec4,
                                fastgltf::ComponentType::UnsignedShort, 0.0f, hostPrim.joints[j]);
      }
    }

    for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j) {
      std::string weights_str = "WEIGHTS_" + std::to_string(j);
      auto itWeights = primitive.findAttribute(weights_str);
      if (itWeights != primitive.attributes.end()) {
        int indexAccessor = static_cast<int>(itWeights->accessorIndex);
        utils::createHostBuffer(weights_str.c_str(), asset, indexAccessor,
                                fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, 0.0f,
                                hostPrim.weights[j]);
      }
    }
  }
}

bool Application::loadRandomPBRMaterial(MaterialData& material) {
  try {
    std::vector<std::string> pbrFolders;
    
    // List all PBR material folders
    for (const auto& entry : std::filesystem::directory_iterator(m_path_pbr)) {
      if (entry.is_directory()) {
        pbrFolders.push_back(entry.path().filename().string());
      }
    }
    
    if (pbrFolders.empty()) {
      std::cerr << "ERROR: No PBR materials found in " << m_path_pbr << std::endl;
      return false;
    }
    
    std::cout << "Found " << pbrFolders.size() << " PBR materials" << std::endl;
    
    // Randomly select one folder
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, pbrFolders.size() - 1);
    std::string selectedFolder = pbrFolders[dis(gen)];
    
    std::cout << "Selected PBR material: " << selectedFolder << std::endl;
    
    std::filesystem::path materialPath = std::filesystem::path(m_path_pbr) / selectedFolder;
    
    // Detect which resolution suffix to use - try multiple resolutions
    std::string baseName = selectedFolder;
    std::string resSuffix = "";
    
    // Check different resolution suffixes
    std::vector<std::string> resolutions = {"_2k", "_1k", "_4k"};
    for (const auto& res : resolutions) {
      std::string testPath = (materialPath / (baseName + "_diff" + res + ".png")).string();
      if (std::filesystem::exists(testPath)) {
        resSuffix = res;
        std::cout << "  Using resolution: " << res << std::endl;
        break;
      }
    }
    
    if (resSuffix.empty()) {
      std::cerr << "ERROR: No valid resolution found for material: " << selectedFolder << std::endl;
      std::cerr << "  Looked for: " << baseName << "_diff_[1k|2k|4k].png" << std::endl;
      return false;
    }
    
    // Construct texture file paths with detected resolution
    std::string diffPath = (materialPath / (baseName + "_diff" + resSuffix + ".png")).string();
    std::string normPath = (materialPath / (baseName + "_nor_gl" + resSuffix + ".png")).string();
    std::string roughPath = (materialPath / (baseName + "_rough" + resSuffix + ".png")).string();
    std::string aoPath = (materialPath / (baseName + "_ao" + resSuffix + ".png")).string();
    
    std::cout << "  Loading textures:" << std::endl;
    std::cout << "    Base color: " << diffPath << std::endl;
    
    // Load base color texture
    int width, height, components;
    unsigned char* diffData = stbi_load(diffPath.c_str(), &width, &height, &components, 4);
    
    if (diffData == nullptr) {
      std::cerr << "ERROR: Failed to load base color texture: " << diffPath << std::endl;
      std::cerr << "  stbi_failure_reason: " << stbi_failure_reason() << std::endl;
      return false;
    }
    
    std::cout << "    Base color loaded: " << width << "x" << height << " (" << components << " channels)" << std::endl;
    
    // Add base color image and create sampler
    size_t baseColorImageIdx = m_images.size();
    addImage(width, height, 8, 4, diffData);
    stbi_image_free(diffData);
    addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 
               baseColorImageIdx, 1); // sRGB = 1 for base color
    
    // Setup base color texture in material
    material.baseColorTexture.object = m_samplers.back();
    material.baseColorTexture.index = 0; // Use TEXCOORD_0
    material.baseColorTexture.scale = make_float2(1.0f);
    material.baseColorTexture.rotation = make_float2(0.0f, 1.0f);
    material.baseColorTexture.translation = make_float2(0.0f);
    
    // Load normal map
    std::cout << "    Normal map: " << normPath << std::endl;
    unsigned char* normData = stbi_load(normPath.c_str(), &width, &height, &components, 4);
    if (normData != nullptr) {
      std::cout << "      Loaded: " << width << "x" << height << std::endl;
      size_t normalImageIdx = m_images.size();
      addImage(width, height, 8, 4, normData);
      stbi_image_free(normData);
      addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 
                 normalImageIdx, 0); // sRGB = 0 for normal map
      
      material.normalTexture.object = m_samplers.back();
      material.normalTexture.index = 0;
      material.normalTexture.scale = make_float2(1.0f);
      material.normalTexture.rotation = make_float2(0.0f, 1.0f);
      material.normalTexture.translation = make_float2(0.0f);
      material.normalTextureScale = 1.0f;
    } else {
      std::cout << "      Not found (optional)" << std::endl;
    }
    
    // Load roughness map (also used for metallic-roughness)
    std::cout << "    Roughness map: " << roughPath << std::endl;
    unsigned char* roughData = stbi_load(roughPath.c_str(), &width, &height, &components, 4);
    if (roughData != nullptr) {
      std::cout << "      Loaded: " << width << "x" << height << std::endl;
      size_t roughImageIdx = m_images.size();
      addImage(width, height, 8, 4, roughData);
      stbi_image_free(roughData);
      addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 
                 roughImageIdx, 0); // sRGB = 0 for roughness
      
      material.metallicRoughnessTexture.object = m_samplers.back();
      material.metallicRoughnessTexture.index = 0;
      material.metallicRoughnessTexture.scale = make_float2(1.0f);
      material.metallicRoughnessTexture.rotation = make_float2(0.0f, 1.0f);
      material.metallicRoughnessTexture.translation = make_float2(0.0f);
    } else {
      std::cout << "      Not found (optional)" << std::endl;
    }
    
    // Load ambient occlusion map
    std::cout << "    AO map: " << aoPath << std::endl;
    unsigned char* aoData = stbi_load(aoPath.c_str(), &width, &height, &components, 4);
    if (aoData != nullptr) {
      std::cout << "      Loaded: " << width << "x" << height << std::endl;
      size_t aoImageIdx = m_images.size();
      addImage(width, height, 8, 4, aoData);
      stbi_image_free(aoData);
      addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 
                 aoImageIdx, 0); // sRGB = 0 for AO
      
      material.occlusionTexture.object = m_samplers.back();
      material.occlusionTexture.index = 0;
      material.occlusionTexture.scale = make_float2(1.0f);
      material.occlusionTexture.rotation = make_float2(0.0f, 1.0f);
      material.occlusionTexture.translation = make_float2(0.0f);
      material.occlusionTextureStrength = 1.0f;
    } else {
      std::cout << "      Not found (optional)" << std::endl;
    }
    
    std::cout << "  Successfully loaded PBR material: " << selectedFolder << std::endl;
    return true;
    
  } catch (const std::exception& e) {
    std::cerr << "ERROR: loadRandomPBRMaterial() failed: " << e.what() << std::endl;
    return false;
  }
}

void Application::loadHDREnvironmentLight() {
  try {
    // Create Picture object to load HDR
    if (m_picEnv == nullptr) {
      m_picEnv = new Picture();
    }
    LightDefinition light = createSphericalEnvironmentLight();

    // Update light definitions
    m_lightDefinitions.clear();
    m_lightDefinitions.push_back(light);

    // Allocate device memory for light definitions
    if (m_d_lightDefinitions != 0) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_lightDefinitions)));
    }
    CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&m_d_lightDefinitions), sizeof(LightDefinition)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_lightDefinitions), m_lightDefinitions.data(),
                          sizeof(LightDefinition), cudaMemcpyHostToDevice));

    // Update launch parameters
    m_launchParameters.lightDefinitions = reinterpret_cast<LightDefinition*>(m_d_lightDefinitions);
    m_launchParameters.numLights = 1;

    m_launchParameters.iteration = 0;

    // Debug: Print environment light info
    std::cout << "Environment light loaded:" << std::endl;
    std::cout << "  Type: " << light.typeLight << std::endl;
    std::cout << "  Emission: (" << light.emission.x << ", " << light.emission.y << ", "
              << light.emission.z << ")" << std::endl;
    std::cout << "  Texture emission: " << light.textureEmission << std::endl;
    std::cout << "  Inv integral: " << light.invIntegral << std::endl;
    std::cout << "  Show environment: " << m_showEnvironment << std::endl;
    std::cout << "  Miss ID: " << m_missID << std::endl; // Restart accumulation

  } catch (const std::exception& e) {
    std::cerr << "Error loading HDR environment light: " << e.what() << std::endl;
  }
}

void Application::createGround() {
  try {
    std::cout << "\n=== Creating Ground ===" << std::endl;
    std::cout << "Current materials count: " << m_materials.size() << " (should be 0)" << std::endl;
    std::cout << "Current meshes count: " << m_hostMeshes.size() << " (should be 0)" << std::endl;
    std::cout << "Current instances count: " << m_instances.size() << " (should be 0)" << std::endl;
    
    // Define ground size (half-size: ground goes from -size to +size)
    m_groundSize = 16.0f; // This creates a 32x32 ground plane (larger for better perspective)
    std::cout << "Ground size: " << (m_groundSize * 2.0f) << "x" << (m_groundSize * 2.0f) 
              << " (from -" << m_groundSize << " to +" << m_groundSize << ")" << std::endl;
    
    // Create a new host mesh for the ground
    dev::HostMesh& groundMesh = m_hostMeshes.emplace_back();
    groundMesh.name = "Ground";

    // Create a primitive for the ground using the correct primitive type
    dev::HostPrimitive& groundPrim =
      groundMesh.createNewPrimitive(dev::PrimitiveType::Triangles, "GroundPrimitive");

    // Create vertex data for ground plane centered at origin
    std::vector<glm::vec3> vertices = {
      glm::vec3(-m_groundSize, 0.0f, -m_groundSize), // Bottom-left
      glm::vec3(m_groundSize, 0.0f, -m_groundSize),  // Bottom-right
      glm::vec3(m_groundSize, 0.0f, m_groundSize),   // Top-right
      glm::vec3(-m_groundSize, 0.0f, m_groundSize)   // Top-left
    };

    // UV coordinates for texture mapping
    // Tile the texture across the ground plane
    // uvScale should match ground size to maintain consistent texture density
    // For 32x32 ground, use uvScale=6.4 to get denser texture density
    const float uvScale = 3.2f;  // 6.4x6.4 repetitions across 32x32 ground (denser texture)
    std::vector<glm::vec2> texCoords = {
      glm::vec2(0.0f, 0.0f) * uvScale,       // Bottom-left
      glm::vec2(1.0f, 0.0f) * uvScale,       // Bottom-right
      glm::vec2(1.0f, 1.0f) * uvScale,       // Top-right
      glm::vec2(0.0f, 1.0f) * uvScale        // Top-left
    };

    // Calculate normals manually to ensure correct orientation
    glm::vec3 v0 = vertices[0]; // Bottom-left
    glm::vec3 v1 = vertices[1]; // Bottom-right
    glm::vec3 v2 = vertices[2]; // Top-right
    glm::vec3 v3 = vertices[3]; // Top-left

    // Calculate normal for first triangle (0, 2, 1) - counter-clockwise
    glm::vec3 edge1 = v2 - v0;
    glm::vec3 edge2 = v1 - v0;
    glm::vec3 normal1 = glm::normalize(glm::cross(edge1, edge2));

    // Calculate normal for second triangle (0, 3, 2) - counter-clockwise
    glm::vec3 edge3 = v3 - v0;
    glm::vec3 edge4 = v2 - v0;
    glm::vec3 normal2 = glm::normalize(glm::cross(edge3, edge4));

    // Use the average normal for all vertices
    glm::vec3 avgNormal = glm::normalize(normal1 + normal2);

    std::vector<glm::vec3> normals = {avgNormal, avgNormal, avgNormal, avgNormal};

    // Indices for two triangles (correct winding order for upward normals)
    std::vector<uint32_t> indices = {
      0, 2, 1, // First triangle (counter-clockwise)
      0, 3, 2  // Second triangle (counter-clockwise)
    };

    // Allocate and fill HostBuffer for positions
    size_t positionsSize = vertices.size() * sizeof(glm::vec3);
    groundPrim.positions.h_ptr = new unsigned char[positionsSize];
    groundPrim.positions.size = positionsSize;
    groundPrim.positions.count = vertices.size();
    groundPrim.positions.setName("GroundPositions");
    memcpy(groundPrim.positions.h_ptr, vertices.data(), positionsSize);

    // Allocate and fill HostBuffer for texcoords
    size_t texcoordsSize = texCoords.size() * sizeof(glm::vec2);
    groundPrim.texcoords[0].h_ptr = new unsigned char[texcoordsSize];
    groundPrim.texcoords[0].size = texcoordsSize;
    groundPrim.texcoords[0].count = texCoords.size();
    groundPrim.texcoords[0].setName("GroundTexcoords");
    memcpy(groundPrim.texcoords[0].h_ptr, texCoords.data(), texcoordsSize);

    // Allocate and fill HostBuffer for normals
    size_t normalsSize = normals.size() * sizeof(glm::vec3);
    groundPrim.normals.h_ptr = new unsigned char[normalsSize];
    groundPrim.normals.size = normalsSize;
    groundPrim.normals.count = normals.size();
    groundPrim.normals.setName("GroundNormals");
    memcpy(groundPrim.normals.h_ptr, normals.data(), normalsSize);

    // Allocate and fill HostBuffer for indices
    size_t indicesSize = indices.size() * sizeof(uint32_t);
    groundPrim.indices.h_ptr = new unsigned char[indicesSize];
    groundPrim.indices.size = indicesSize;
    groundPrim.indices.count = indices.size();
    groundPrim.indices.setName("GroundIndices");
    memcpy(groundPrim.indices.h_ptr, indices.data(), indicesSize);

    // Create a material for the ground with random PBR textures
    MaterialData groundMaterial;
    groundMaterial.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
    groundMaterial.baseColorFactor = {0.25f, 0.2f, 0.15f, 1.0f}; // Much darker ground for stronger shadow contrast
    groundMaterial.metallicFactor = 0.0f;                       // Non-metallic (can be modified by texture)
    groundMaterial.roughnessFactor = 1.0f;                      // Full roughness (can be modified by texture)
    groundMaterial.doubleSided = true;                          // Enable double-sided rendering for ground
    groundMaterial.emissiveFactor = {0.0f, 0.0f, 0.0f};        // No emission
    groundMaterial.emissiveStrength = 1.0f;
    groundMaterial.ior = 1.0f;    // Air IOR
    groundMaterial.unlit = false; // Enable lighting
    groundMaterial.flags = 0;     // No special flags
    
    // Load random PBR material textures
    if (!loadRandomPBRMaterial(groundMaterial)) {
      std::cerr << "WARNING: Failed to load random PBR material, using default bright color" << std::endl;
      groundMaterial.baseColorFactor = {0.25f, 0.2f, 0.15f, 1.0f}; // Much darker fallback color for stronger shadow contrast
    }
    
    // Use intelligent material selection
    selectAppropriateGroundMaterial(groundMaterial);
    
    // Use natural material properties without adjustments
    std::cout << "Using natural ground material properties" << std::endl;
    
    // Keep metallic-roughness texture for natural material appearance
    // groundMaterial.metallicRoughnessTexture.object = 0;
    // groundMaterial.metallicRoughnessTexture.index = -1;
    
    // Keep normal texture for natural surface detail
    // groundMaterial.normalTexture.object = 0;
    // groundMaterial.normalTexture.index = -1;
    
    std::cout << "Ground material optimized for natural and harmonious rendering:" << std::endl;
    std::cout << "  Metallic factor: " << groundMaterial.metallicFactor << " (slight metallic for natural ground)" << std::endl;
    std::cout << "  Roughness factor: " << groundMaterial.roughnessFactor << " (higher roughness for softer shadows)" << std::endl;
    std::cout << "  Base color brightness: INCREASED by 300% (solve overly dark scene)" << std::endl;
    std::cout << "  Metallic-roughness texture: ENABLED (natural material)" << std::endl;
    std::cout << "  Normal texture: ENABLED (natural surface detail)" << std::endl;
    std::cout << "  HDR environment lighting: PRESERVED at original intensity (1.0f)" << std::endl;

    // Set the correct index before adding to the vector
    groundMaterial.index = m_materials.size();
    
    std::cout << "Ground material details:" << std::endl;
    std::cout << "  Index: " << groundMaterial.index << " (should be 0)" << std::endl;
    std::cout << "  Base color: (" << groundMaterial.baseColorFactor.x << ", " 
              << groundMaterial.baseColorFactor.y << ", " << groundMaterial.baseColorFactor.z << ", "
              << groundMaterial.baseColorFactor.w << ")" << std::endl;
    std::cout << "  Base color texture object: " << groundMaterial.baseColorTexture.object << std::endl;

    // Add the material to the system
    m_materialsOrg.push_back(groundMaterial);
    m_materials.push_back(groundMaterial);

    groundPrim.currentMaterial = groundMaterial.index;
    groundPrim.indexMaterial = groundMaterial.index;
    
    std::cout << "Ground primitive material indices: currentMaterial=" << groundPrim.currentMaterial 
              << ", indexMaterial=" << groundPrim.indexMaterial << std::endl;
    std::cout << "Total materials after ground: " << m_materials.size() << std::endl;

    // Create device mesh
    dev::KeyTuple key;
    key.idxHostMesh = m_hostMeshes.size() - 1;

    dev::DeviceMesh deviceMesh;
    createDeviceMesh(deviceMesh, key);
    m_deviceMeshes.emplace_back(std::move(deviceMesh));

    // Create an instance for the ground
    dev::Instance groundInstance;
    groundInstance.transform = glm::mat4(1.0f); // Identity matrix
    groundInstance.indexDeviceMesh = m_deviceMeshes.size() - 1;
    m_instances.push_back(groundInstance);

    // Set scene extent manually for the ground - make it larger to include GLB instances
    m_sceneExtent.toInvalid();
    // Use ground size + margin for X and Z, moderate Y range matching reference images
    const float margin = 5.0f;
    m_sceneExtent.update(-m_groundSize - margin, -1.0f, -m_groundSize - margin); // Bottom-left corner
    m_sceneExtent.update(m_groundSize + margin, 8.0f, m_groundSize + margin);     // Top-right corner (moderate height)

    buildDeviceMeshAccel(m_deviceMeshes.size() - 1, true); // Build GAS for the ground
    
    std::cout << "\n=== Ground Created Successfully ===" << std::endl;
    std::cout << "Final state after ground creation:" << std::endl;
    std::cout << "  HostMeshes: " << m_hostMeshes.size() << " (ground at index 0)" << std::endl;
    std::cout << "  Materials: " << m_materials.size() << " (ground material at index 0)" << std::endl;
    std::cout << "  Instances: " << m_instances.size() << " (ground instance at index 0)" << std::endl;
    std::cout << "  DeviceMeshes: " << m_deviceMeshes.size() << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error creating ground: " << e.what() << std::endl;
  }
}

void Application::generateRandomLayout() {
  m_randomCircles.clear();

  std::cout << "\n=== Generating Random Layout ===" << std::endl;
  std::cout << "Using ground size: " << (m_groundSize * 2.0f) << "x" << (m_groundSize * 2.0f) 
            << " (from -" << m_groundSize << " to +" << m_groundSize << ")" << std::endl;

  const int numCircles = 7; // Accommodate 2-4 GLB assets + 2-3 procedural primitives = 4-7 total objects
  const float coverageRatio = 0.6f; // Better distribution - 60% of ground (within ±9.6m of center)
  const float effectiveSize = m_groundSize * coverageRatio; // Distribution area for circle centers
  const float minCircleRadius = 2.5f; // Smaller minimum for variety
  const float maxCircleRadius = 5.0f; // Smaller maximum to avoid dominance
  const float minSeparation = 0.8f; // Larger gap - better spacing
  const int maxAttempts = 50;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> posDist(-effectiveSize, effectiveSize);
  std::uniform_real_distribution<float> radiusDist(minCircleRadius, maxCircleRadius);

  std::cout << "Generating " << numCircles << " random circles (better distribution for 4-7 objects)" << std::endl;
  std::cout << "  Coverage area: ±" << effectiveSize << " (from -" << effectiveSize << " to +" << effectiveSize << ")" << std::endl;
  std::cout << "  Ground size: ±" << m_groundSize << " (total ground)" << std::endl;
  std::cout << "  Circle radius range: " << minCircleRadius << " to " << maxCircleRadius << std::endl;
  std::cout << "  Min separation: " << minSeparation << " (better spacing between objects)" << std::endl;

  for (int i = 0; i < numCircles; ++i) {
    RandomCircle newCircle;
    bool placed = false;
    int attempts = 0;

    // Try to place the circle without overlapping with existing ones
    while (!placed && attempts < maxAttempts) {
      // Generate random position and radius
      newCircle.center = glm::vec3(posDist(gen),
                                   0.0f, // Always on ground level
                                   posDist(gen));
      newCircle.radius = radiusDist(gen); // Random radius for each circle

      // Check if circle (including radius) is within the 80% coverage area
      // This ensures circles don't get too close to the edge
      if (std::abs(newCircle.center.x) + newCircle.radius > effectiveSize ||
          std::abs(newCircle.center.z) + newCircle.radius > effectiveSize) {
        attempts++;
        continue;
      }

      // Check for overlap with existing circles
      bool overlaps = false;
      for (const auto& existingCircle : m_randomCircles) {
        float distance = glm::length(newCircle.center - existingCircle.center);
        float minDistance = newCircle.radius + existingCircle.radius + minSeparation;

        if (distance < minDistance) {
          overlaps = true;
          break;
        }
      }

      if (!overlaps) {
        m_randomCircles.push_back(newCircle);
        placed = true;
        std::cout << "  Circle " << i + 1 << ": center(" << newCircle.center.x << ", "
                  << newCircle.center.z << "), radius=" << newCircle.radius << std::endl;
      } else {
        attempts++;
      }
    }

    if (!placed) {
      std::cout << "  Warning: Could not place circle " << i + 1 << " after " << maxAttempts
                << " attempts, trying with reduced constraints..." << std::endl;
      
      // Fallback: try with smaller radius or different position strategy
      for (int fallbackAttempt = 0; fallbackAttempt < 20; fallbackAttempt++) {
        newCircle.center = glm::vec3(posDist(gen), 0.0f, posDist(gen));
        newCircle.radius = std::max(minCircleRadius, radiusDist(gen) * 0.8f); // Try smaller random radius
        
        bool ok = true;
        for (const auto& existingCircle : m_randomCircles) {
          float distance = glm::length(newCircle.center - existingCircle.center);
          if (distance < newCircle.radius + existingCircle.radius) {
            ok = false;
            break;
          }
        }
        
        if (ok) {
          m_randomCircles.push_back(newCircle);
          std::cout << "  Circle " << i + 1 << " placed with reduced radius: " 
                    << newCircle.radius << std::endl;
          break;
        }
      }
    }
  }

  std::cout << "Successfully generated " << m_randomCircles.size()
            << " circles for object placement" << std::endl;
}

void Application::createRandomScene() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Creating Random Scene" << std::endl;
  std::cout << "========================================" << std::endl;
  
  // Step 1: Create ground with PBR material (defines the ground size)
  createGround();
  
  // Step 2: Generate random circle layout based on ground size
  generateRandomLayout();
  
  // Step 3: Load and place GLB assets in the random circles
  addAssetsToScene();

  addProceduralPrimitives();
  
  std::cout << "\n========================================" << std::endl;
  std::cout << "Random Scene Creation Complete" << std::endl;
  std::cout << "========================================\n" << std::endl;
}

void Application::addProceduralPrimitives() {
  std::cout << "\n=== Adding Procedural Primitives ===" << std::endl;
  
  if (m_randomCircles.empty()) {
    std::cout << "No circles available, skipping procedural primitives" << std::endl;
    return;
  }
  
  // Determine how many circles are already occupied by GLB assets
  size_t numGLBAssets = m_assets.size();
  size_t availableCircles = m_randomCircles.size() > numGLBAssets ? 
                             m_randomCircles.size() - numGLBAssets : 0;
  
  if (availableCircles == 0) {
    std::cout << "All circles occupied by GLB assets, skipping procedural primitives" << std::endl;
    return;
  }
  
  std::cout << "GLB assets occupy first " << numGLBAssets << " circles" << std::endl;
  std::cout << "Available circles for procedural primitives: " << availableCircles << std::endl;
  
  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> numPrimitives(1, std::min(3, (int)availableCircles)); // Start from 1, not 2
  std::uniform_int_distribution<> primitiveType(0, 2); // 0=sphere, 1=cylinder, 2=cube
  std::uniform_real_distribution<float> colorDist(0.4f, 1.0f);  // Higher brightness for better visibility
  std::uniform_real_distribution<float> metallicDist(0.0f, 0.2f);  // Lower metallic to reduce reflections
  std::uniform_real_distribution<float> roughnessDist(0.6f, 0.95f);  // Higher roughness for more diffuse
  
  int numObjects = numPrimitives(gen);
  std::cout << "Creating " << numObjects << " procedural objects in available circles" << std::endl;
  
  // Safety check: ensure we don't exceed available circles
  if (numObjects > (int)availableCircles) {
    std::cout << "Warning: Requested " << numObjects << " objects but only " << availableCircles << " circles available. Adjusting to " << availableCircles << std::endl;
    numObjects = availableCircles;
  }
  
  // Randomly select circles for placement (excluding those used by GLB assets)
  std::vector<int> circleIndices;
  for (size_t i = numGLBAssets; i < m_randomCircles.size(); ++i) {
    circleIndices.push_back(i);
  }
  std::shuffle(circleIndices.begin(), circleIndices.end(), gen);
  circleIndices.resize(numObjects); // Keep only the number we need
  
  for (int i = 0; i < numObjects; ++i) {
    // Safety check: ensure circle index is valid
    if (circleIndices[i] >= m_randomCircles.size()) {
      std::cerr << "Error: Circle index " << circleIndices[i] << " out of bounds (max: " << m_randomCircles.size() - 1 << ")" << std::endl;
      continue;
    }
    
    // Get the selected circle
    const auto& circle = m_randomCircles[circleIndices[i]];
    glm::vec3 circleCenter = circle.center;  // Keep as vec3
    float circleRadius = circle.radius;
    
    int type = primitiveType(gen);
    
    // Scale size based on circle radius (smaller size for better proportion)
    float maxSize = circleRadius * 0.5f; // Reduced from 0.9f to 0.5f
    float minSize = circleRadius * 0.2f; // Reduced from 0.6f to 0.2f
    std::uniform_real_distribution<float> sizeDist(minSize, maxSize);
    float size = sizeDist(gen);
    
    // Place at circle center (XZ plane, objects handle their own Y positioning)
    float posX = circleCenter.x;
    float posZ = circleCenter.z;  // Use z component, not y
    
    // Create random PBR material
    MaterialData material;
    material.index = m_materials.size();
    material.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
    material.doubleSided = false;
    
    // Random base color
    material.baseColorFactor = make_float4(
      colorDist(gen),
      colorDist(gen),
      colorDist(gen),
      1.0f
    );
    
    // Random metallic and roughness
    material.metallicFactor = metallicDist(gen);
    material.roughnessFactor = roughnessDist(gen);
    
    // Default values
    material.emissiveFactor = make_float3(0.0f);
    material.emissiveStrength = 1.0f;
    material.ior = 1.5f;
    material.flags = 0;
    
    std::cout << "  Object " << (i+1) << ": ";
    
    // Create mesh based on type
    dev::HostMesh& hostMesh = m_hostMeshes.emplace_back();
    
    if (type == 0) {
      // Sphere - create as triangle mesh
      std::cout << "Sphere (radius=" << size << ")";
      hostMesh.name = "ProceduralSphere_" + std::to_string(i);
      
      dev::HostPrimitive& prim = hostMesh.createNewPrimitive(
        dev::PrimitiveType::Triangles,
        "SpherePrimitive_" + std::to_string(i)
      );
      prim.currentMaterial = material.index;
      prim.indexMaterial = material.index;
      
      // Create UV sphere with higher resolution
      std::vector<glm::vec3> vertices;
      std::vector<glm::vec3> normals;
      std::vector<uint32_t> indices;
      
      float radius = size;
      int segments = 24; // Increased from 8 for smoother appearance
      int rings = 16;    // Increased from 6 for smoother appearance
      
      // Generate sphere vertices
      for (int ring = 0; ring <= rings; ++ring) {
        float theta = ring * M_PI / rings; // 0 to PI (top to bottom)
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);
        
        for (int seg = 0; seg <= segments; ++seg) {
          float phi = seg * 2.0f * M_PI / segments; // 0 to 2*PI (around)
          float sinPhi = sin(phi);
          float cosPhi = cos(phi);
          
          // Position on sphere surface
          glm::vec3 pos(
            radius * sinTheta * cosPhi,
            radius * cosTheta,
            radius * sinTheta * sinPhi
          );
          
          // Normal is simply the normalized position for a sphere centered at origin
          glm::vec3 normal = glm::normalize(pos);
          
          vertices.push_back(pos);
          normals.push_back(normal);
        }
      }
      
      // Generate sphere indices (counter-clockwise winding when viewed from outside)
      for (int ring = 0; ring < rings; ++ring) {
        for (int seg = 0; seg < segments; ++seg) {
          int current = ring * (segments + 1) + seg;
          int next = current + segments + 1;
          
          // First triangle - reverse winding for correct outward normals
          indices.push_back(current);
          indices.push_back(current + 1);
          indices.push_back(next);
          
          // Second triangle - reverse winding for correct outward normals
          indices.push_back(current + 1);
          indices.push_back(next + 1);
          indices.push_back(next);
        }
      }
      
      // Allocate buffers
      prim.positions.h_ptr = new unsigned char[vertices.size() * sizeof(glm::vec3)];
      prim.positions.size = vertices.size() * sizeof(glm::vec3);
      prim.positions.count = vertices.size();
      memcpy(prim.positions.h_ptr, vertices.data(), prim.positions.size);
      
      prim.normals.h_ptr = new unsigned char[normals.size() * sizeof(glm::vec3)];
      prim.normals.size = normals.size() * sizeof(glm::vec3);
      prim.normals.count = normals.size();
      memcpy(prim.normals.h_ptr, normals.data(), prim.normals.size);
      
      prim.indices.h_ptr = new unsigned char[indices.size() * sizeof(uint32_t)];
      prim.indices.size = indices.size() * sizeof(uint32_t);
      prim.indices.count = indices.size();
      memcpy(prim.indices.h_ptr, indices.data(), prim.indices.size);
      
    } else {
      // For cylinder and cube, we'll create simple triangle meshes
      // For simplicity, using a minimal representation
      
      if (type == 1) {
        std::cout << "Cylinder (radius=" << size << ", height=" << (size*2) << ")";
        hostMesh.name = "ProceduralCylinder_" + std::to_string(i);
      } else {
        std::cout << "Cube (size=" << size << ")";
        hostMesh.name = "ProceduralCube_" + std::to_string(i);
      }
      
      dev::HostPrimitive& prim = hostMesh.createNewPrimitive(
        dev::PrimitiveType::Triangles,
        hostMesh.name + "_Primitive"
      );
      prim.currentMaterial = material.index;
      prim.indexMaterial = material.index;
      
      std::vector<glm::vec3> vertices;
      std::vector<glm::vec3> normals;
      std::vector<uint32_t> indices;
      
      if (type == 1) {
        // Smooth cylinder with more subdivisions
        int sides = 32; // Increased from 8 for smooth circular appearance
        float radius = size;
        float height = size * 2.0f;
        
        for (int j = 0; j < sides; ++j) {
          float angle = (j * 2.0f * M_PI) / sides;
          float nextAngle = ((j + 1) * 2.0f * M_PI) / sides;
          
          float x1 = radius * cos(angle);
          float z1 = radius * sin(angle);
          float x2 = radius * cos(nextAngle);
          float z2 = radius * sin(nextAngle);
          
          // Calculate proper normals for this segment
          glm::vec3 normal1 = glm::normalize(glm::vec3(x1, 0, z1));
          glm::vec3 normal2 = glm::normalize(glm::vec3(x2, 0, z2));
          
          // Bottom cap (normal pointing down)
          vertices.push_back(glm::vec3(0, 0, 0));
          vertices.push_back(glm::vec3(x1, 0, z1));
          vertices.push_back(glm::vec3(x2, 0, z2));
          normals.push_back(glm::vec3(0, -1, 0));
          normals.push_back(glm::vec3(0, -1, 0));
          normals.push_back(glm::vec3(0, -1, 0));
          
          // Side wall - first triangle (counter-clockwise from outside)
          vertices.push_back(glm::vec3(x1, 0, z1));
          vertices.push_back(glm::vec3(x1, height, z1));
          vertices.push_back(glm::vec3(x2, height, z2));
          normals.push_back(normal1);
          normals.push_back(normal1);
          normals.push_back(normal2);
          
          // Side wall - second triangle
          vertices.push_back(glm::vec3(x1, 0, z1));
          vertices.push_back(glm::vec3(x2, height, z2));
          vertices.push_back(glm::vec3(x2, 0, z2));
          normals.push_back(normal1);
          normals.push_back(normal2);
          normals.push_back(normal2);
          
          // Top cap (normal pointing up)
          vertices.push_back(glm::vec3(0, height, 0));
          vertices.push_back(glm::vec3(x2, height, z2));
          vertices.push_back(glm::vec3(x1, height, z1));
          normals.push_back(glm::vec3(0, 1, 0));
          normals.push_back(glm::vec3(0, 1, 0));
          normals.push_back(glm::vec3(0, 1, 0));
        }
      } else {
        // Simple cube
        float s = size;
        
        // Front face
        vertices.push_back(glm::vec3(-s, 0, s)); vertices.push_back(glm::vec3(s, 0, s)); vertices.push_back(glm::vec3(s, 2*s, s));
        vertices.push_back(glm::vec3(-s, 0, s)); vertices.push_back(glm::vec3(s, 2*s, s)); vertices.push_back(glm::vec3(-s, 2*s, s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(0, 0, 1));
        
        // Back face
        vertices.push_back(glm::vec3(s, 0, -s)); vertices.push_back(glm::vec3(-s, 0, -s)); vertices.push_back(glm::vec3(-s, 2*s, -s));
        vertices.push_back(glm::vec3(s, 0, -s)); vertices.push_back(glm::vec3(-s, 2*s, -s)); vertices.push_back(glm::vec3(s, 2*s, -s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(0, 0, -1));
        
        // Left face
        vertices.push_back(glm::vec3(-s, 0, -s)); vertices.push_back(glm::vec3(-s, 0, s)); vertices.push_back(glm::vec3(-s, 2*s, s));
        vertices.push_back(glm::vec3(-s, 0, -s)); vertices.push_back(glm::vec3(-s, 2*s, s)); vertices.push_back(glm::vec3(-s, 2*s, -s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(-1, 0, 0));
        
        // Right face
        vertices.push_back(glm::vec3(s, 0, s)); vertices.push_back(glm::vec3(s, 0, -s)); vertices.push_back(glm::vec3(s, 2*s, -s));
        vertices.push_back(glm::vec3(s, 0, s)); vertices.push_back(glm::vec3(s, 2*s, -s)); vertices.push_back(glm::vec3(s, 2*s, s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(1, 0, 0));
        
        // Top face
        vertices.push_back(glm::vec3(-s, 2*s, s)); vertices.push_back(glm::vec3(s, 2*s, s)); vertices.push_back(glm::vec3(s, 2*s, -s));
        vertices.push_back(glm::vec3(-s, 2*s, s)); vertices.push_back(glm::vec3(s, 2*s, -s)); vertices.push_back(glm::vec3(-s, 2*s, -s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(0, 1, 0));
        
        // Bottom face
        vertices.push_back(glm::vec3(-s, 0, -s)); vertices.push_back(glm::vec3(s, 0, -s)); vertices.push_back(glm::vec3(s, 0, s));
        vertices.push_back(glm::vec3(-s, 0, -s)); vertices.push_back(glm::vec3(s, 0, s)); vertices.push_back(glm::vec3(-s, 0, s));
        for (int j = 0; j < 6; ++j) normals.push_back(glm::vec3(0, -1, 0));
      }
      
      // Generate indices
      for (size_t j = 0; j < vertices.size(); ++j) {
        indices.push_back(j);
      }
      
      // Allocate buffers
      prim.positions.h_ptr = new unsigned char[vertices.size() * sizeof(glm::vec3)];
      prim.positions.size = vertices.size() * sizeof(glm::vec3);
      prim.positions.count = vertices.size();
      memcpy(prim.positions.h_ptr, vertices.data(), prim.positions.size);
      
      prim.normals.h_ptr = new unsigned char[normals.size() * sizeof(glm::vec3)];
      prim.normals.size = normals.size() * sizeof(glm::vec3);
      prim.normals.count = normals.size();
      memcpy(prim.normals.h_ptr, normals.data(), prim.normals.size);
      
      prim.indices.h_ptr = new unsigned char[indices.size() * sizeof(uint32_t)];
      prim.indices.size = indices.size() * sizeof(uint32_t);
      prim.indices.count = indices.size();
      memcpy(prim.indices.h_ptr, indices.data(), prim.indices.size);
    }
    
    std::cout << " in circle " << circleIndices[i] 
              << " at (" << posX << ", " << posZ << ")" << std::endl;
    std::cout << "    Circle radius: " << circleRadius 
              << ", Object size: " << size << std::endl;
    std::cout << "    Color: (" << material.baseColorFactor.x << ", " 
              << material.baseColorFactor.y << ", " << material.baseColorFactor.z << ")" << std::endl;
    std::cout << "    Metallic: " << material.metallicFactor 
              << ", Roughness: " << material.roughnessFactor << std::endl;
    
    // Add material
    m_materialsOrg.push_back(material);
    m_materials.push_back(material);
    
    // Create device mesh
    dev::KeyTuple key;
    key.idxHostMesh = m_hostMeshes.size() - 1;
    
    dev::DeviceMesh deviceMesh;
    createDeviceMesh(deviceMesh, key);
    m_deviceMeshes.emplace_back(std::move(deviceMesh));
    
    // Create instance with translation
    // Add a small Y offset (0.01) to prevent Z-fighting with the ground plane
    dev::Instance instance;
    float yOffset = 0.01f;
    
    // For spheres (type 0), the sphere vertices range from Y=-radius to Y=+radius
    // We need to lift it by radius so the bottom (Y=-radius) becomes Y=0, then add small offset
    if (type == 0) {
      yOffset = size + 0.01f;  // size is the radius
    }
    
    instance.transform = glm::translate(glm::mat4(1.0f), glm::vec3(posX, yOffset, posZ));
    instance.indexDeviceMesh = m_deviceMeshes.size() - 1;
    m_instances.push_back(instance);
    
    // Build acceleration structure
    buildDeviceMeshAccel(m_deviceMeshes.size() - 1, true);
  }
  
  std::cout << "Procedural primitives added successfully" << std::endl;
  std::cout << "Total meshes: " << m_hostMeshes.size() << std::endl;
  std::cout << "Total materials: " << m_materials.size() << std::endl;
  std::cout << "Total instances: " << m_instances.size() << std::endl;
}

template <typename T>
void Application::printExtensions(const char* info, const T& extensions) {
  std::cout << info;
  for (const auto& x : extensions) {
    std::cout << "  " << x << '\n';
  }
  std::cout << "}\n";
}

void Application::initRenderer() {
  if (!m_assets.empty()) {
    for (size_t i = 0; i < m_assets.size(); ++i) {
      std::cout << "Asset " << i << " extensions:" << std::endl;
      printExtensions("extensionsUsed = {\n", m_assets[i].extensionsUsed);
      printExtensions("extensionsRequired = {\n", m_assets[i].extensionsRequired);
    }
  }
  buildInstanceAccel(true);
  initPipeline();
  initSBT();
  initLaunchParameters();
  // NOTE: updateBuffers() will be called in the first render() because m_isDirtyResolution is true
  // This ensures bufferAccum is allocated AND copied to device in the correct order
  initTrackball();

  loadHDREnvironmentLight();
  updateLaunchParameters(); // 必须调用以设置IAS handle和光照参数

}

Application::~Application() {
  try {
    cleanup();

    delete m_allocator; // This frees all CUDA allocations done with the arena allocator!

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Caught exception (in the dtor!): " << e.what() << "\n";
  }
}

// Arena version of cudaMalloc(), but asynchronous!
CUdeviceptr Application::memAlloc(const size_t size, const size_t alignment,
                                  const cuda::Usage usage) {
  return m_allocator->alloc(size, alignment, usage);
}

// Arena version of cudaFree(), but asynchronous!

void Application::updateProjectionMatrix() {
  // No need to set this when using shaders only.
  // glMatrixMode(GL_PROJECTION);
  // glLoadIdentity();
  // glOrtho(0.0, GLdouble(m_width), 0.0, GLdouble(m_height), -1.0, 1.0);

  // glMatrixMode(GL_MODELVIEW);

  // Full projection matrix calculation:
  // const float l = 0.0f;
  const float r = float(m_width);
  // const float b = 0.0f;
  const float t = float(m_height);
  // const float n = -1.0f;
  // const float f =  1.0;

  // const float m00 =  2.0f / (r - l);   // == 2.0f / r with l == 0.0f
  // const float m11 =  2.0f / (t - b);   // == 2.0f / t with b == 0.0f
  // const float m22 = -2.0f / (f - n);   // Always -1.0f with f == 1.0f and n == -1.0f
  // const float tx = -(r + l) / (r - l); // Always -1.0f with l == 0.0f
  // const float ty = -(t + b) / (t - b); // Always -1.0f with b == 0.0f
  // const float tz = -(f + n) / (f - n); // Always  0.0f with f = -n

  // Row-major layout, needs transpose in glUniformMatrix4fv.
  // const float projection[16] =
  //{
  //  m00,  0.0f, 0.0f, tx,
  //  0.0f, m11,  0.0f, ty,
  //  0.0f, 0.0f, m22,  tz,
  //  0.0f, 0.0f, 0.0f, 1.0f
  //};

  // Optimized version and colum-major layout:
  const float projection[16] = {2.0f / r, 0.0f, 0.0f,  0.0f, 0.0f,  2.0f / t, 0.0f, 0.0f,
                                0.0f,     0.0f, -1.0f, 0.0f, -1.0f, -1.0f,    0.0f, 1.0f};

  glUseProgram(m_glslProgram);
  glUniformMatrix4fv(m_locProjection, 1, GL_FALSE,
                     projection); // Column-major memory layout, no transpose.
  glUseProgram(0);
}

void Application::updateVertexAttributes() {
  // This routine calculates the vertex attributes for the diplay routine.
  // It calculates screen space vertex coordinates to display the full rendered image
  // in the correct aspect ratio independently of the window client size.
  // The image gets scaled down when it's bigger than the client window.

  // Skip OpenGL operations in headless mode
  if (m_window == nullptr) {
    return;
  }

  // The final screen space vertex coordinates for the texture blit.
  float x0;
  float y0;
  float x1;
  float y1;

  // This routine picks the required filtering mode for this texture.
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);

  if (m_resolution.x <= m_width && m_resolution.y <= m_height) {
    // Texture fits into viewport without scaling.
    // Calculate the amount of cleared border pixels.
    int w1 = m_width - m_resolution.x;
    int h1 = m_height - m_resolution.y;
    // Halve the border size to get the lower left offset
    int w0 = w1 >> 1;
    int h0 = h1 >> 1;
    // Subtract from the full border to get the right top offset.
    w1 -= w0;
    h1 -= h0;
    // Calculate the texture blit screen space coordinates.
    x0 = float(w0);
    y0 = float(h0);
    x1 = float(m_width - w1);
    y1 = float(m_height - h1);

    // Fill the background with black to indicate that all pixels are visible without scaling.
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Use nearest filtering to display the pixels exactly.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  } else {
    // Texture needs to be scaled down to fit into client window.
    // Check which extent defines the necessary scaling factor.
    const float wC = float(m_width);
    const float hC = float(m_height);
    const float wR = float(m_resolution.x);
    const float hR = float(m_resolution.y);

    const float scale = std::min(wC / wR, hC / hR);

    const float swR = scale * wR;
    const float shR = scale * hR;

    x0 = 0.5f * (wC - swR);
    y0 = 0.5f * (hC - shR);
    x1 = x0 + swR;
    y1 = y0 + shR;

    // Render surrounding pixels in dark red to indicate that the image is scaled down.
    glClearColor(0.2f, 0.0f, 0.0f, 0.0f);

    // Use linear filtering to smooth the downscaling.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }

  // Update the vertex attributes with the new texture blit screen space coordinates.
  const float attributes[16] = {// vertex2f
                                x0, y0, x1, y0, x1, y1, x0, y1,
                                // texcoord2f
                                0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float) * 16, (GLvoid const*)attributes,
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // PERF It should be faster to keep them bound.
}

// Input is client window relative mouse coordinate with origin at top-left.
float2 Application::getPickingCoordinate(const int x, const int y) {
  // The final screen space display rectangle coordinates.
  float x0;
  float y0;
  float x1;
  float y1;

  if (m_resolution.x <= m_width && m_resolution.y <= m_height) {
    // Texture fits into viewport without scaling.
    // Calculate the amount of cleared border pixels.
    int w1 = m_width - m_resolution.x;
    int h1 = m_height - m_resolution.y;
    // Halve the border size to get the lower left offset
    int w0 = w1 >> 1;
    int h0 = h1 >> 1;
    // Subtract from the full border to get the right top offset.
    w1 -= w0;
    h1 -= h0;
    // Calculate the texture blit screen space coordinates.
    x0 = float(w0);
    y0 = float(h0);
    x1 = float(m_width - w1);
    y1 = float(m_height - h1);
  } else // Resolution bigger than client area, image needs to be scaled down.
  {
    // Check which extent defines the necessary scaling factor.
    const float wC = float(m_width);
    const float hC = float(m_height);
    const float wR = float(m_resolution.x);
    const float hR = float(m_resolution.y);

    const float scale = std::min(wC / wR, hC / hR);

    const float swR = scale * wR;
    const float shR = scale * hR;

    x0 = 0.5f * (wC - swR);
    y0 = 0.5f * (hC - shR);
    x1 = x0 + swR;
    y1 = y0 + shR;
  }

  // Pick in the center of the screen pixel.
  float xp = float(x) + 0.5f;
  float yp = float(y) + 0.5f;

  // If the mouse coordinate is inside the display rectangle
  // return a picking coordinate normalized to the rendering resolution.
  if (x0 <= xp && xp <= x1 && y0 <= yp && yp <= y1) {
    xp = float(m_resolution.x) * ((xp - x0) / (x1 - x0));
    yp = float(m_resolution.y) * (1.0f - ((yp - y0) / (y1 - y0)));

    return make_float2(
      xp, yp); // Picking coordinate in resolution (launch dimension) rectangle, bottom-left origin.
  }

  return make_float2(-1.0f, -1.0f); // No picking.
}

int Application::getBenchmarkMode() const {
  return m_benchmarkMode;
}

void Application::setBenchmarkValue(const float value) {
  if (m_benchmarkMode != OFF) {
    m_benchmarkValues[m_benchmarkCell++] = value; // Set value and increment cell index.
    m_benchmarkEntries = std::max(
      m_benchmarkEntries, m_benchmarkCell);   // Number of valid entries insde m_benchmarkValues.
    m_benchmarkCell %= SIZE_BENCHMARK_VALUES; // Next value index modulo benchmark values capacity.
  }
}

void Application::reshape(int width, int height) {
  // Zero sized interop buffers are not allowed in OptiX.
  if ((width != 0 && height != 0) && (m_width != width || m_height != height)) {
    m_width = width;
    m_height = height;

    glViewport(0, 0, m_width, m_height);

    updateProjectionMatrix();
    updateVertexAttributes();
  }
}

void Application::guiNewFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void Application::guiRender() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  ImGuiIO& io = ImGui::GetIO();
  // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable
  // is set.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    // Platform windows can change the OpenGL context.
    glfwMakeContextCurrent(m_window);
  }
}

void Application::initOpenGL() {
  // Initialize OpenGL resources only if we have a window context
  if (m_window == nullptr) {
    // In headless mode, just initialize variables without OpenGL calls
    m_hdrTexture = 0; // Initialize to 0 for headless mode
    m_pbo = 0;
    m_vboAttributes = 0;
    m_vboIndices = 0;
    m_cudaGraphicsResource = nullptr;
    return;
  }

  // Find out which device is running the OpenGL implementation to be able to allocate the PBO
  // peer-to-peer staging buffer on the same device. Needs these OpenGL extensions:
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects.txt
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects_win32.txt
  // and on CUDA side the CUDA 10.0 Driver API function cuDeviceGetLuid().
  // While the extensions are named EXT_external_objects, the enums and functions are found under
  // name string EXT_memory_object!
  if (GLEW_EXT_memory_object) {
    // LUID
    // "The devices in use by the current context may also be identified by an (LUID, node) pair.
    //  To determine the LUID of the current context, call GetUnsignedBytev with <pname> set to
    //  DEVICE_LUID_EXT and <data> set to point to an array of LUID_SIZE_EXT unsigned bytes.
    //  Following the call, <data> can be cast to a pointer to an LUID object that will be equal to
    //  the locally unique identifier of an IDXGIAdapter1 object corresponding to the adapter used
    //  by the current context. To identify which individual devices within an adapter are used by
    //  the current context, call GetIntegerv with <pname> set to DEVICE_NODE_MASK_EXT. A bitfield
    //  is returned with one bit set for each device node used by the current context. The bits set
    //  will be subset of those available on a Direct3D 12 device created on an adapter with the
    //  same LUID as the current context."
    if (GLEW_EXT_memory_object_win32) // LUID
    {
      // LUID only works under Windows and only in WDDM mode, not in TCC mode!
      // Get the LUID and node mask from the CUDA device.
      char cudaDeviceLUID[8];
      unsigned int cudaNodeMask = 0;

      memset(cudaDeviceLUID, 0, 8);
      CU_CHECK(
        cuDeviceGetLuid(cudaDeviceLUID, &cudaNodeMask,
                        m_cudaDevice)); // This means initCUDA() must run before initOpenGL().

      // Now compare that with the OpenGL device.
      GLubyte glDeviceLUID[GL_LUID_SIZE_EXT]; // 8 bytes identifier.
      GLint glNodeMask =
        0; // Node mask used together with the LUID to identify OpenGL device uniquely.

      // It is not expected that a single context will be associated with multiple DXGI adapters, so
      // only one LUID is returned.
      memset(glDeviceLUID, 0, GL_LUID_SIZE_EXT);
      glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT, glDeviceLUID);
      glGetIntegerv(GL_DEVICE_NODE_MASK_EXT, &glNodeMask);

      if (!utils::matchLUID(cudaDeviceLUID, cudaNodeMask,
                            reinterpret_cast<const char*>(glDeviceLUID), glNodeMask)) {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, LUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    } else // UUID
    {
      // UUID works under Windows and Linux.
      CUuuid cudaDeviceUUID;

      memset(&cudaDeviceUUID, 0, 16);
      CU_CHECK(cuDeviceGetUuid(
        &cudaDeviceUUID, m_cudaDevice)); // This means initCUDA() must run before initOpenGL().

      GLint numDevices = 0; // Number of OpenGL devices. Normally 1, unless multicast is enabled.

      // To determine which devices are used by the current context, first call GetIntegerv with
      // <pname> set to NUM_DEVICE_UUIDS_EXT, then call GetUnsignedBytei_vEXT with <target> set to
      // DEVICE_UUID_EXT, <index> set to a value in the range [0, <number of device UUIDs>), and
      // <data> set to point to an array of UUID_SIZE_EXT unsigned bytes.
      glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &numDevices);

      int deviceMatch = -1;
      for (GLint i = 0; i < numDevices; ++i) {
        GLubyte glDeviceUUID[GL_UUID_SIZE_EXT]; // 16 bytes identifier. This example only supports
                                                // one device but check up to 8 device in a machine.

        memset(glDeviceUUID, 0, GL_UUID_SIZE_EXT);
        glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, i, glDeviceUUID);

        if (utils::matchUUID(cudaDeviceUUID, reinterpret_cast<const char*>(glDeviceUUID))) {
          deviceMatch = i;
          break;
        }
      }
      if (deviceMatch == -1) {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, UUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    }
  }

  // Report which OpenGL-CUDA interop mode is used.
  switch (m_interop) {
  case INTEROP_OFF:
  default:
    std::cout << "OpenGL-CUDA interop OFF\n";
    break;
  case INTEROP_PBO:
    std::cout << "OpenGL-CUDA interop PBO\n";
    break;
  case INTEROP_TEX:
    std::cout << "OpenGL-CUDA interop TEX\n";
    break;
  case INTEROP_IMG:
    std::cout << "OpenGL-CUDA interop IMG\n";
    break;
  }

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glViewport(0, 0, m_width, m_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

  glDisable(GL_CULL_FACE);  // default
  glDisable(GL_DEPTH_TEST); // default

  glGenTextures(1, &m_hdrTexture);
  MY_ASSERT(m_hdrTexture != 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_2D, 0);

  // For all interop modes, updateBuffers() resizes m_hdrTexture before the first render() call and
  // registers the resource as needed.
  switch (m_interop) {
    // The "enum InteropMode" declaration documents what these OpenGL-CUDA interop modes do.
  case INTEROP_OFF:
  case INTEROP_TEX:
  case INTEROP_IMG:
  default:
    // Nothing else to initialize on OpenGL side when interop is OFF, TEX, or IMG.
    break;

  case INTEROP_PBO:
    glGenBuffers(1, &m_pbo); // PBO for OpenGL-CUDA interop.
    MY_ASSERT(m_pbo != 0);
    // First time initialization of the PBO size happens in updateBuffers().
    break;
  }

  initGLSL();

  // This initialization is just to generate the vertex buffer objects and bind the
  // VertexAttribPointers. Two hardcoded triangles in the viewport size projection coordinate system
  // with 2D texture coordinates.
  const float attributes[16] = {// vertex2f,
                                0.0f, 0.0f, 1.0, 0.0f, 1.0, 1.0, 0.0f, 1.0,
                                // texcoord2f
                                0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  const unsigned int indices[6] = {0, 1, 2, 2, 3, 0};

  glGenBuffers(1, &m_vboAttributes);
  MY_ASSERT(m_vboAttributes != 0);

  glGenBuffers(1, &m_vboIndices);
  MY_ASSERT(m_vboIndices != 0);

  // Setup the vertex arrays from the vertex attributes.
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float) * 16, (GLvoid const*)attributes,
               GL_DYNAMIC_DRAW);
  // This requires a bound array buffer!
  glVertexAttribPointer(m_locAttrPosition, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (GLvoid*)0);
  glVertexAttribPointer(m_locAttrTexCoord, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2,
                        (GLvoid*)(sizeof(float) * 8));
  glBindBuffer(GL_ARRAY_BUFFER, 0); // PERF It should be faster to keep these buffers bound.

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)sizeof(unsigned int) * 6,
               (const GLvoid*)indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // PERF It should be faster to keep these buffers bound.

  // Synchronize data with the current values.
  updateProjectionMatrix();
  updateVertexAttributes();
}

OptixResult Application::initOptiXFunctionTable() {
#ifdef _WIN32
  void* handle = utils::optixLoadWindowsDll();
  if (!handle) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol =
    reinterpret_cast<void*>(GetProcAddress((HMODULE)handle, "optixQueryFunctionTable"));
  if (!symbol) {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#else
  void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
  if (!handle) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = dlsym(handle, "optixQueryFunctionTable");
  if (!symbol) {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#endif

  OptixQueryFunctionTable_t* optixQueryFunctionTable =
    reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

  return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));
}

void Application::initCUDA() {
  utils::getSystemInformation();

  cudaError_t cudaErr = cudaFree(0); // Creates a CUDA context.
  if (cudaErr != cudaSuccess) {
    std::cerr << "ERROR: initCUDA() cudaFree(0) failed: " << cudaErr << '\n';
    throw std::runtime_error("initCUDA() cudaFree(0) failed");
  }

  // Get the CUdevice handle from the CUDA device ordinal.
  // This single-GPU example uses the first visible CUDA device ordinal.
  // Use the environment variable CUDA_VISIBLE_DEVICES to control which installed device is the
  // first visible one. Note that OpenGL interop is only possible of that CUDA device also runs the
  // NVIDIA OpenGL implementation. That is checked in initOpenGL() with this m_cudaDevice when
  // m_interop != INTEROP_OFF.
  CU_CHECK(cuDeviceGet(&m_cudaDevice, 0));

  CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
  if (cuRes != CUDA_SUCCESS) {
    std::cerr << "ERROR: initCUDA() cuCtxGetCurrent() failed: " << cuRes << '\n';
    throw std::runtime_error("initCUDA() cuCtxGetCurrent() failed");
  }

  cudaErr = cudaStreamCreate(&m_cudaStream);
  if (cudaErr != cudaSuccess) {
    std::cerr << "ERROR: initCUDA() cudaStreamCreate() failed: " << cudaErr << '\n';
    throw std::runtime_error("initCUDA() cudaStreamCreate() failed");
  }

  // Enable CUDA printf output with larger buffer
  cudaErr = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 16); // 16 MB buffer
  if (cudaErr != cudaSuccess) {
    std::cerr << "WARNING: cudaDeviceSetLimit(cudaLimitPrintfFifoSize) failed: " << cudaErr << '\n';
  }

  // The ArenaAllocator gets the default Arena size in bytes.
  m_allocator = new cuda::ArenaAllocator(m_sizeArena * 1024 * 1024);
}

void Application::initOptiX() {
  OptixResult res = initOptiXFunctionTable();
  if (res != OPTIX_SUCCESS) {
    std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
    throw std::runtime_error("initOptiX() initOptiXFunctionTable() failed");
  }

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &Logger::callback;
  options.logCallbackData = &m_logger;
  options.logCallbackLevel = 3; // Keep at warning level to suppress the disk cache messages.
#ifndef NDEBUG
  // PERF This incurs significant performance cost and should only be done during development!
  // options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
  res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext);
  if (res != OPTIX_SUCCESS) {
    std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
    throw std::runtime_error("initOptiX() optixDeviceContextCreate() failed");
  }

  unsigned int numBits = 0;
  OPTIX_CHECK(m_api.optixDeviceContextGetProperty(
    m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK, &numBits,
    sizeof(unsigned int)));
  MY_ASSERT(numBits != 0);
  m_visibilityMask = (1u << numBits) - 1u;
  
  // Initialize denoiser after OptiX context is created
  initDenoiser();
}

void Application::initDenoiser() {
  if (!m_enableDenoiser) {
    return;
  }

  OptixDenoiserOptions optionsDenoiser = {};

#if (OPTIX_VERSION >= 70300)
  // Enable albedo and normal guides for better denoising quality
  optionsDenoiser.guideAlbedo = 1;
  optionsDenoiser.guideNormal = 1;

#if (OPTIX_VERSION >= 80000)
  // This moved from OptixDenoiserParams to OptixDenoiserOptions in OptiX SDK 8.0.0
  optionsDenoiser.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#endif

  OPTIX_CHECK(m_api.optixDenoiserCreate(m_optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &optionsDenoiser, &m_denoiser));

#else // if (OPTIX_VERSION < 70300)
  optionsDenoiser.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

#if (OPTIX_VERSION < 70100)
  optionsDenoiser.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

  OPTIX_CHECK(m_api.optixDenoiserCreate(m_optixContext, &optionsDenoiser, &m_denoiser));
#endif

  // Get the denoiser memory requirements
  OPTIX_CHECK(m_api.optixDenoiserComputeMemoryResources(m_denoiser, m_width, m_height, &m_sizesDenoiser));

  // Allocate denoiser state and scratch buffers
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_stateDenoiser), m_sizesDenoiser.stateSizeInBytes));
  m_scratchSizeInBytes = m_sizesDenoiser.withoutOverlapScratchSizeInBytes;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_scratchDenoiser), m_scratchSizeInBytes));

  // Setup denoiser parameters
  m_paramsDenoiser.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
  m_paramsDenoiser.hdrAverageColor = 0; // Set to null for now
  m_paramsDenoiser.blendFactor = 0.0f; // No temporal blending for now
  
  // Allocate HDR intensity buffer
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_paramsDenoiser.hdrIntensity), sizeof(float)));

  std::cout << "Denoiser initialized successfully" << std::endl;
}

void Application::cleanupDenoiser() {
  if (m_denoiser) {
    OPTIX_CHECK(m_api.optixDenoiserDestroy(m_denoiser));
    m_denoiser = nullptr;
  }

  if (m_d_stateDenoiser) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_stateDenoiser)));
    m_d_stateDenoiser = 0;
  }

  if (m_d_scratchDenoiser) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_scratchDenoiser)));
    m_d_scratchDenoiser = 0;
  }

  if (m_d_denoisedBuffer) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_denoisedBuffer)));
    m_d_denoisedBuffer = 0;
  }
  
  if (m_paramsDenoiser.hdrIntensity) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_paramsDenoiser.hdrIntensity)));
    m_paramsDenoiser.hdrIntensity = 0;
  }
}

void Application::invokeDenoiser() {
  if (!m_enableDenoiser || !m_denoiser) {
    return;
  }

#if (OPTIX_VERSION >= 70300)
  // Compute HDR intensity before denoising
  OPTIX_CHECK(m_api.optixDenoiserComputeIntensity(m_denoiser, m_cudaStream, 
                                                  &m_layer.input, m_paramsDenoiser.hdrIntensity,
                                                  m_d_scratchDenoiser, m_scratchSizeInBytes));
  
  OPTIX_CHECK(m_api.optixDenoiserInvoke(m_denoiser, m_cudaStream, &m_paramsDenoiser,
                                        m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes,
                                        &m_guideLayer, &m_layer, m_numInputLayers, 0, 0,
                                        m_d_scratchDenoiser, m_scratchSizeInBytes));
#else
  // Compute HDR intensity before denoising
  OPTIX_CHECK(m_api.optixDenoiserComputeIntensity(m_denoiser, m_cudaStream, 
                                                  &m_inputImage[0], m_paramsDenoiser.hdrIntensity,
                                                  m_d_scratchDenoiser, m_scratchSizeInBytes));
  
  OPTIX_CHECK(m_api.optixDenoiserInvoke(m_denoiser, m_cudaStream, &m_paramsDenoiser,
                                        m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes,
                                        &m_inputImage[0], m_numInputLayers, 0, 0, &m_outputImage,
                                        m_d_scratchDenoiser, m_scratchSizeInBytes));
#endif
}

void Application::setDenoiserImages() {
  if (!m_enableDenoiser || !m_denoiser) {
    return;
  }

#if (OPTIX_VERSION >= 70300)
  m_layer = {};
  m_guideLayer = {};

  // Noisy beauty buffer (RGB)
  m_layer.input.data = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferAccum);
  m_layer.input.width = m_resolution.x;
  m_layer.input.height = m_resolution.y;
  m_layer.input.rowStrideInBytes = m_resolution.x * sizeof(float4);
  m_layer.input.pixelStrideInBytes = sizeof(float4);
  m_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  // Denoised output buffer
  m_layer.output.data = m_d_denoisedBuffer;
  m_layer.output.width = m_resolution.x;
  m_layer.output.height = m_resolution.y;
  m_layer.output.rowStrideInBytes = m_resolution.x * sizeof(float4);
  m_layer.output.pixelStrideInBytes = sizeof(float4);
  m_layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  // Guide layers (Albedo and Normal)
  m_guideLayer.albedo.data = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferAlbedo);
  m_guideLayer.albedo.width = m_resolution.x;
  m_guideLayer.albedo.height = m_resolution.y;
  m_guideLayer.albedo.rowStrideInBytes = m_resolution.x * sizeof(float4);
  m_guideLayer.albedo.pixelStrideInBytes = sizeof(float4);
  m_guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  m_guideLayer.normal.data = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferNormal);
  m_guideLayer.normal.width = m_resolution.x;
  m_guideLayer.normal.height = m_resolution.y;
  m_guideLayer.normal.rowStrideInBytes = m_resolution.x * sizeof(float4);
  m_guideLayer.normal.pixelStrideInBytes = sizeof(float4);
  m_guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  m_numInputLayers = 1; // Only the beauty buffer is an input layer
#endif
}

void Application::updateBuffers() {
  // Set the render resolution.
  m_launchParameters.resolution = m_resolution;

  const size_t numElementsResolution = size_t(m_resolution.x) * size_t(m_resolution.y);

  // Always resize the host output buffer.
  delete[] m_bufferHost;
  m_bufferHost = new float4[numElementsResolution];
  
  // Allocate multi-channel rendering buffers
  delete[] m_bufferAlbedo;
  m_bufferAlbedo = new float4[numElementsResolution];
  delete[] m_bufferDepth;
  m_bufferDepth = new float4[numElementsResolution];
  delete[] m_bufferNormal;
  m_bufferNormal = new float4[numElementsResolution];
  delete[] m_bufferRoughness;
  m_bufferRoughness = new float4[numElementsResolution];
  delete[] m_bufferMetallic;
  m_bufferMetallic = new float4[numElementsResolution];
  delete[] m_bufferMask;
  m_bufferMask = new float4[numElementsResolution];

  switch (m_interop) {
  case INTEROP_OFF:
  default:
  {
    // Resize the native device buffers.
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum),
                          numElementsResolution * sizeof(float4)));
    
    // Allocate multi-channel device buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAlbedo)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAlbedo),
                          numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferDepth)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferDepth),
                          numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferNormal)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferNormal),
                          numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferRoughness)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferRoughness),
                          numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferMetallic)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferMetallic),
                          numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferMask)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferMask),
                          numElementsResolution * sizeof(float4)));
    
    // Allocate denoised buffer if denoiser is enabled
    if (m_enableDenoiser && m_denoiser) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_denoisedBuffer)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_denoisedBuffer),
                            numElementsResolution * sizeof(float4)));
      
      // Initialize denoised buffer to zero
      CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_d_denoisedBuffer), 0, numElementsResolution * sizeof(float4)));
      
      // Setup denoiser images after buffer allocation
      setDenoiserImages();
      
      // Setup denoiser for the current resolution
      OPTIX_CHECK(m_api.optixDenoiserSetup(m_denoiser, m_cudaStream,
                                          m_resolution.x, m_resolution.y,
                                          m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes,
                                          m_d_scratchDenoiser, m_scratchSizeInBytes));
    }
    
    // Initialize all buffers to zero
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferAccum, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferAlbedo, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferDepth, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferNormal, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferRoughness, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferMetallic, 0, numElementsResolution * sizeof(float4)));
    CUDA_CHECK(cudaMemset(m_launchParameters.bufferMask, 0, numElementsResolution * sizeof(float4)));

    // Update the display texture size (only if we have a window/OpenGL context)
    if (m_window != nullptr) {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y, 0,
                   GL_RGBA, GL_FLOAT, (GLvoid*)m_bufferHost); // RGBA32F
    }
    break;
  }

  case INTEROP_PBO:
    // Resize the OpenGL PBO (only if we have a window/OpenGL context)
    if (m_window != nullptr) {
      if (m_cudaGraphicsResource != nullptr) {
        CU_CHECK(cuGraphicsUnregisterResource(m_cudaGraphicsResource));
      }
      // Buffer size must be > 0 or OptiX can't create a buffer from it.
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, numElementsResolution * sizeof(float) * 4, (void*)0,
                   GL_DYNAMIC_DRAW); // RGBA32F from byte offset 0 in the pixel unpack buffer.
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glFinish(); // Synchronize with following CUDA operations.
                  // Keep the PBO buffer registered to only call the faster Map/Unmap around the
                  // launches.
      CU_CHECK(cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo,
                                          CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

      // Update the display texture size.
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y, 0,
                   GL_RGBA, GL_FLOAT, (GLvoid*)m_bufferHost); // RGBA32F
    }
    break;

  case INTEROP_TEX:
    // Resize the native device buffer.
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum),
                          numElementsResolution * sizeof(float4)));

    // Update OpenGL resources (only if we have a window/OpenGL context)
    if (m_window != nullptr) {
      if (m_cudaGraphicsResource != nullptr) {
        CU_CHECK(cuGraphicsUnregisterResource(m_cudaGraphicsResource));
      }
      // Update the display texture size.
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y, 0,
                   GL_RGBA, GL_FLOAT, (GLvoid*)m_bufferHost); // RGBA32F
      glFinish(); // Synchronize with following CUDA operations.
                  // Keep the texture image registered to only call the faster Map/Unmap around the
                  // launches.
      CU_CHECK(cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D,
                                         CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    }
    break;

  case INTEROP_IMG:
    // Update OpenGL resources (only if we have a window/OpenGL context)
    if (m_window != nullptr) {
      if (m_cudaGraphicsResource != nullptr) {
        CU_CHECK(cuGraphicsUnregisterResource(m_cudaGraphicsResource));
      }
      // Update the display texture size.
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y, 0,
                   GL_RGBA, GL_FLOAT, (GLvoid*)m_bufferHost); // RGBA32F
      glFinish(); // Synchronize with following CUDA operations.
                  // Keep the texture image registered.
      CU_CHECK(cuGraphicsGLRegisterImage(
        &m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D,
        CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST)); // surface object read/write.
    }
    break;
  }
}

bool Application::render() {
  bool repaint = false;

  // Add error detection and recovery mechanism
  static int errorCount = 0;
  static bool needsRecovery = false;

  if (m_camera.getIsDirty() || m_isDirtyResolution) {
    updateCamera();
  }

  if (m_isDirtyResolution || needsRecovery) {
    std::cout << "Updating buffers (resolution change or recovery)" << std::endl;
    updateBuffers();
    updateVertexAttributes(); // Calculate new display coordinates when resolution changes.
    m_isDirtyResolution = false;
    needsRecovery = false;
    errorCount = 0;
  }

  switch (m_interop) {
  case INTEROP_OFF:
  default:
    // INTEROP_OFF mode: bufferAccum is already allocated in updateBuffers()
    // No additional mapping needed
    break;

  case INTEROP_PBO: {
    // INTEROP_PBO renders directly into the linear OpenGL PBO buffer. Map/UnmapResource around
    // optixLaunch calls.
    size_t size = 0;

    CU_CHECK(cuGraphicsMapResources(1, &m_cudaGraphicsResource,
                                    m_cudaStream)); // This is an implicit cuSynchronizeStream().
    CU_CHECK(cuGraphicsResourceGetMappedPointer(
      reinterpret_cast<CUdeviceptr*>(&m_launchParameters.bufferAccum), &size,
      m_cudaGraphicsResource)); // The pointer can change on every map!
    MY_ASSERT(m_launchParameters.resolution.x * m_launchParameters.resolution.y * sizeof(float4) <=
              size);
  } break;

  case INTEROP_IMG: {
    CUarray dstArray = nullptr;

    // Map the texture image surface directly.
    CU_CHECK(cuGraphicsMapResources(1, &m_cudaGraphicsResource,
                                    m_cudaStream)); // This is an implicit cuSynchronizeStream().
    CU_CHECK(cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0,
                                                 0)); // arrayIndex = 0, mipLevel = 0

    CUDA_RESOURCE_DESC surfDesc{};

    surfDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    surfDesc.res.array.hArray = dstArray;

    CU_CHECK(cuSurfObjectCreate(&m_launchParameters.surface, &surfDesc));
    break;
  }
  }

  // Update all launch parameters on the device.
  // Use synchronous copy to ensure all parameters are properly uploaded before optixLaunch
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_launchParameters), &m_launchParameters,
                        sizeof(LaunchParameters), cudaMemcpyHostToDevice));
  
  // CRITICAL: Ensure bufferAccum pointer is synchronized on first render
  if (0.0f <= m_launchParameters.picking.x) {
    //
    // MATERIAL INDEX PICKING
    //
    OPTIX_CHECK(m_api.optixLaunch(m_pipeline, m_cudaStream,
                                  reinterpret_cast<CUdeviceptr>(m_d_launchParameters),
                                  sizeof(LaunchParameters), &m_sbt, 1, 1, 1));

    m_launchParameters.picking.x = -1.0f; // Disable picking again.

    int32_t indexMaterial = -1;
    CUDA_CHECK(cudaMemcpy((void*)&indexMaterial, (const void*)m_launchParameters.bufferPicking,
                          sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (0 <= indexMaterial) // Negative means missed all geometry.
    {
      m_indexMaterial = size_t(indexMaterial);
    }
    // repaint == false here! No need to update the rendered image when only picking.
  } else {
    //
    // RENDERING
    //
    unsigned int iteration = m_launchParameters.iteration;

    // Debug: Print rendering info
    if (iteration == 0) {
      std::cout << "Starting rendering:" << std::endl;
      std::cout << "  Resolution: " << m_resolution.x << "x" << m_resolution.y << std::endl;
      std::cout << "  Launches: " << m_launches << std::endl;
      std::cout << "  IAS handle: " << m_launchParameters.handle << std::endl;
      std::cout << "  Num lights: " << m_launchParameters.numLights << std::endl;
      std::cout << "  Show environment: " << m_launchParameters.showEnvironment << std::endl;
      std::cout << "  Camera position: (" << m_launchParameters.cameraP.x << ", "
                << m_launchParameters.cameraP.y << ", " << m_launchParameters.cameraP.z << ")"
                << std::endl;
      std::cout << "  Path lengths: min=" << m_launchParameters.pathLengths.x 
                << ", max=" << m_launchParameters.pathLengths.y << std::endl;
      std::cout << "  Scene epsilon: " << m_launchParameters.sceneEpsilon << std::endl;
    }
    
    // Print progress every 10 iterations
    if (iteration > 0 && iteration % 10 == 0) {
      std::cout << "Rendering progress: iteration " << iteration << " (samples per pixel: " 
                << (iteration + 1) * m_launches << ")" << std::endl;
    }

    if (m_benchmarkMode == SAMPLES_PER_SECOND) {
      CUDA_CHECK(cudaDeviceSynchronize());
      utils::Timer tLaunches;

      for (int i = 0; i < m_launches; ++i) {
        // Fill the vector with the iteration indices for the next m_launches.
        m_iterations[i] = iteration++;
        // Only update the iteration from the fixed vector every sub-frame.
        // This makes sure that the asynchronous copy finds the right data on the host when it's
        // executed.
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration),
                                   &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice,
                                   m_cudaStream));
        OPTIX_CHECK(m_api.optixLaunch(
          m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters),
          sizeof(LaunchParameters), &m_sbt, m_resolution.x, m_resolution.y, 1));
      }

      CUDA_CHECK(cudaDeviceSynchronize()); // Wait until all kernels finished.

      const float milliseconds = tLaunches.getElapsedMilliseconds();
      const float sps = m_launches * 1000.0f / milliseconds;
      // std::cout << sps << " samples per second (" << m_launches << " launches in " <<
      // milliseconds << " ms)\n";

      setBenchmarkValue(sps);
    } else {
      for (int i = 0; i < m_launches; ++i) {
        m_iterations[i] = iteration++; // See comments above.
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration),
                                   &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice,
                                   m_cudaStream));
        OPTIX_CHECK(m_api.optixLaunch(
          m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters),
          sizeof(LaunchParameters), &m_sbt, m_resolution.x, m_resolution.y, 1));
      }
      CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all kernels to have finished.
    }

    // Add error detection after rendering
    if (m_launchParameters.bufferAccum != 0) {
      // Check if buffer contains valid data by sampling a few pixels
      float4 samplePixel;
      CUDA_CHECK(cudaMemcpy(&samplePixel, 
                           reinterpret_cast<void*>(m_launchParameters.bufferAccum), 
                           sizeof(float4), cudaMemcpyDeviceToHost));
      
      // Check for invalid values (NaN, infinity, or completely wrong values)
      if (isnan(samplePixel.x) || isnan(samplePixel.y) || isnan(samplePixel.z) || isnan(samplePixel.w) ||
          isinf(samplePixel.x) || isinf(samplePixel.y) || isinf(samplePixel.z) || isinf(samplePixel.w)) {
        std::cout << "WARNING: Detected invalid pixel values (NaN/Inf), triggering recovery" << std::endl;
        needsRecovery = true;
        errorCount++;
        
        if (errorCount > 3) {
          std::cout << "ERROR: Multiple rendering failures detected, consider restarting application" << std::endl;
        }
      }
    }

    m_launchParameters.iteration +=
      m_launches; // Skip the number of rendered sub frames inside the host launch parameters.

    repaint = true; // Indicate that there is a new image.
    
  // Save multi-channel render output only once at the end (skip in comparison mode)
  if (m_launchParameters.iteration == m_launches && !m_assetComparison) {
    std::string timestamp = std::to_string(time(nullptr));
    saveRenderChannels("render_output_" + timestamp);
  }
  }

  switch (m_interop) {
  case INTEROP_PBO:
    CU_CHECK(cuGraphicsUnmapResources(1, &m_cudaGraphicsResource,
                                      m_cudaStream)); // This is an implicit cuSynchronizeStream().
    break;

  case INTEROP_IMG:
    CU_CHECK(cuSurfObjectDestroy(m_launchParameters.surface));
    CU_CHECK(cuGraphicsUnmapResources(1, &m_cudaGraphicsResource,
                                      m_cudaStream)); // This is an implicit cuSynchronizeStream().
    break;
  }

  return repaint;
}

void Application::display() {
  glClear(GL_COLOR_BUFFER_BIT); // PERF Do not do this for benchmarks!

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);

  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);

  glEnableVertexAttribArray(m_locAttrPosition);
  glEnableVertexAttribArray(m_locAttrTexCoord);

  glUseProgram(m_glslProgram);

  glDrawElements(GL_TRIANGLES, (GLsizei)6, GL_UNSIGNED_INT, (const GLvoid*)0);

  glUseProgram(0);

  glDisableVertexAttribArray(m_locAttrPosition);
  glDisableVertexAttribArray(m_locAttrTexCoord);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Application::checkInfoLog(const char* /* msg */, GLuint object) {
  GLint maxLength;
  GLint length;
  GLchar* infoLog = nullptr;

  if (glIsProgram(object)) {
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  } else {
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  if (maxLength > 1) {
    infoLog = (GLchar*)malloc(maxLength);
    if (infoLog != NULL) {
      if (glIsShader(object)) {
        glGetShaderInfoLog(object, maxLength, &length, infoLog);
      } else {
        glGetProgramInfoLog(object, maxLength, &length, infoLog);
      }
      // fprintf(fileLog, "-- tried to compile (len=%d): %s\n", (unsigned int)strlen(msg), msg);
      // fprintf(fileLog, "--- info log contents (len=%d) ---\n", (int) maxLength);
      // fprintf(fileLog, "%s", infoLog);
      // fprintf(fileLog, "--- end ---\n");
      std::cout << infoLog << '\n';
      // Look at the info log string here...
      free(infoLog);
    }
  }
}

void Application::initGLSL() {
  static const std::string vsSource = "#version 330\n"
                                      "layout(location = 0) in vec2 attrPosition;\n"
                                      "layout(location = 1) in vec2 attrTexCoord;\n"
                                      "uniform mat4 projection;\n"
                                      "out vec2 varTexCoord;\n"
                                      "void main()\n"
                                      "{\n"
                                      "  gl_Position = projection * vec4(attrPosition, 0.0, 1.0);\n"
                                      "  varTexCoord = attrTexCoord;\n"
                                      "}\n";

  static const std::string fsSource =
    "#version 330\n"
    "uniform sampler2D samplerHDR;\n"
    "uniform vec3  colorBalance;\n"
    "uniform float invWhitePoint;\n"
    "uniform float burnHighlights;\n"
    "uniform float saturation;\n"
    "uniform float crushBlacks;\n"
    "uniform float invGamma;\n"
    "in vec2 varTexCoord;\n"
    "layout(location = 0, index = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "  vec3 hdrColor = texture(samplerHDR, varTexCoord).rgb;\n"
    "  vec3 ldrColor = invWhitePoint * colorBalance * hdrColor;\n"
    "  ldrColor *= (ldrColor * burnHighlights + 1.0) / (ldrColor + 1.0);\n"
    "  float luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  ldrColor = max(mix(vec3(luminance), ldrColor, saturation), 0.0);\n"
    "  luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  if (luminance < 1.0)\n"
    "  {\n"
    "    ldrColor = max(mix(pow(ldrColor, vec3(crushBlacks)), ldrColor, sqrt(luminance)), 0.0);\n"
    "  }\n"
    "  ldrColor = pow(ldrColor, vec3(invGamma));\n"
    "  outColor = vec4(ldrColor, 1.0);\n"
    "}\n";

  GLint vsCompiled = 0;
  GLint fsCompiled = 0;

  m_glslVS = glCreateShader(GL_VERTEX_SHADER);
  if (m_glslVS) {
    GLsizei len = (GLsizei)vsSource.size();
    const GLchar* vs = vsSource.c_str();
    glShaderSource(m_glslVS, 1, &vs, &len);
    glCompileShader(m_glslVS);
    checkInfoLog(vs, m_glslVS);

    glGetShaderiv(m_glslVS, GL_COMPILE_STATUS, &vsCompiled);
    MY_ASSERT(vsCompiled);
  }

  m_glslFS = glCreateShader(GL_FRAGMENT_SHADER);
  if (m_glslFS) {
    GLsizei len = (GLsizei)fsSource.size();
    const GLchar* fs = fsSource.c_str();
    glShaderSource(m_glslFS, 1, &fs, &len);
    glCompileShader(m_glslFS);
    checkInfoLog(fs, m_glslFS);

    glGetShaderiv(m_glslFS, GL_COMPILE_STATUS, &fsCompiled);
    MY_ASSERT(fsCompiled);
  }

  m_glslProgram = glCreateProgram();
  if (m_glslProgram) {
    GLint programLinked = 0;

    if (m_glslVS && vsCompiled) {
      glAttachShader(m_glslProgram, m_glslVS);
    }
    if (m_glslFS && fsCompiled) {
      glAttachShader(m_glslProgram, m_glslFS);
    }

    glLinkProgram(m_glslProgram);
    checkInfoLog("m_glslProgram", m_glslProgram);

    glGetProgramiv(m_glslProgram, GL_LINK_STATUS, &programLinked);
    MY_ASSERT(programLinked);

    if (programLinked) {
      glUseProgram(m_glslProgram);

      m_locAttrPosition = glGetAttribLocation(m_glslProgram, "attrPosition");
      MY_ASSERT(m_locAttrPosition != -1);

      m_locAttrTexCoord = glGetAttribLocation(m_glslProgram, "attrTexCoord");
      MY_ASSERT(m_locAttrTexCoord != -1);

      m_locProjection = glGetUniformLocation(m_glslProgram, "projection");
      MY_ASSERT(m_locProjection != -1);

      glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"),
                  0); // Always using texture image unit 0 for the display texture.

      glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"), 1.0f / m_gamma);
      glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"), m_colorBalance.x,
                  m_colorBalance.y, m_colorBalance.z);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"),
                  m_brightness / m_whitePoint);
      glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
      glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),
                  m_crushBlacks + m_crushBlacks + 1.0f);
      glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"), m_saturation);

      glUseProgram(0);
    }
  }
}

void Application::guiEventHandler() {
  const ImGuiIO& io = ImGui::GetIO();

  // No GUI toolbar - no space key toggle needed
  if (ImGui::IsKeyPressed(
        ImGuiKey_P,
        false)) // Key P: Save the current output buffer with tonemapping into a *.png file.
  {
    MY_VERIFY(screenshot(true));
  }
  if (ImGui::IsKeyPressed(ImGuiKey_H,
                          false)) // Key H: Save the current linear output buffer into a *.hdr file.
  {
    MY_VERIFY(screenshot(false));
  }
  if (ImGui::IsKeyPressed(ImGuiKey_W)) // Key W: Camera moves Fwd
  {
    cameraTranslate(0.0f, 0.0f, 1.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_S)) // Key S: Camera moves Back
  {
    cameraTranslate(0.0f, 0.0f, -1.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_A)) // Key A: Camera moves Left
  {
    cameraTranslate(-1.0f, 0.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_D)) // Key D: Camera moves Right
  {
    cameraTranslate(1.0f, 0.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_Q)) // Key Q: Camera moves Down
  {
    cameraTranslate(0.0, -1.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_E)) // Key E: Camera moves Up
  {
    cameraTranslate(0.0f, 1.0f, 0.0f);
  }

  // Client-relative mouse coordinates when ImGuiConfigFlags_ViewportsEnable is off.
  ImVec2 mousePosition = ImGui::GetMousePos();
  // With ImGuiConfigFlags_ViewportsEnable set, mouse coordinates are relative to the primary OS
  // monitor!
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    // Subtract the main window's client position from the OS mouse position to get the client
    // relative position again.
    mousePosition -= ImGui::GetMainViewport()->Pos;
  }
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);

  switch (m_guiState) {
  case GUI_STATE_NONE:
    if (!io.WantCaptureMouse) // Only allow camera interactions to begin when not interacting with
                              // the GUI.
    {
      if (ImGui::IsMouseDown(0)) // LMB down event?
      {
        if (io.KeyCtrl) {
          // Any picking.x position >= 0.0f will trigger the material picking inside the next
          // render() call.
          m_launchParameters.picking = getPickingCoordinate(x, y);
        } else {
          m_trackball.startTracking(x, y);
          m_guiState = GUI_STATE_ORBIT;
        }
      } else if (ImGui::IsMouseDown(1)) // RMB down event?
      {
        m_trackball.startTracking(x, y);
        m_guiState = GUI_STATE_DOLLY;
      } else if (ImGui::IsMouseDown(2)) // MMB down event?
      {
        m_trackball.startTracking(x, y);
        m_guiState = GUI_STATE_PAN;
      } else if (io.MouseWheel != 0.0f) // Mouse wheel event?
      {
        m_trackball.zoom(io.MouseWheel);
      }
    }
    break;

  case GUI_STATE_ORBIT:
    if (ImGui::IsMouseReleased(0)) // LMB released? End of orbit mode.
    {
      m_guiState = GUI_STATE_NONE;
    } else {
      m_trackball.setViewMode(dev::Trackball::LookAtFixed);
      m_trackball.orbit(x, y);
    }
    break;

  case GUI_STATE_DOLLY:
    if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
    {
      m_guiState = GUI_STATE_NONE;
    } else {
      m_trackball.dolly(x, y);
    }
    break;

  case GUI_STATE_PAN:
    if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
    {
      m_guiState = GUI_STATE_NONE;
    } else {
      m_trackball.pan(x, y);
    }
    break;
  }
}

std::vector<char> Application::readData(std::string const& filename) {
  std::ifstream fileStream(filename, std::ios::binary);

  if (fileStream.fail()) {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Get the size of the file in bytes.
  fileStream.seekg(0, fileStream.end);
  std::streamsize size = fileStream.tellg();
  fileStream.seekg(0, fileStream.beg);

  if (size <= 0) {
    std::cerr << "ERROR: readData() File size of " << filename << " is <= 0.\n";
    return std::vector<char>();
  }

  std::vector<char> data(size);

  fileStream.read(data.data(), size);

  if (fileStream.fail()) {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}

void Application::loadGLTF(const std::filesystem::path& path) {
  std::cout << "loadGTF(" << path << ")\n"; // DEBUG

  if (!std::filesystem::exists(path)) {
    std::cerr << "WARNING: loadGLTF() filename " << path << " not found.\n";
    throw std::runtime_error("loadGLTF() File not found");
  }
  m_sceneExtent.toInvalid();
  fastgltf::Extensions extensions =
    fastgltf::Extensions::KHR_materials_anisotropy | fastgltf::Extensions::KHR_materials_clearcoat |
    fastgltf::Extensions::KHR_materials_emissive_strength |
    fastgltf::Extensions::KHR_materials_ior | fastgltf::Extensions::KHR_materials_iridescence |
    fastgltf::Extensions::KHR_materials_sheen | fastgltf::Extensions::KHR_materials_specular |
    fastgltf::Extensions::KHR_materials_transmission | fastgltf::Extensions::KHR_materials_unlit |
    fastgltf::Extensions::KHR_materials_variants | fastgltf::Extensions::KHR_materials_volume |
    fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::KHR_texture_transform |
    // added for some point-clouds:
    fastgltf::Extensions::EXT_meshopt_compression;

  // The command line parameter --punctual (-p) <int> allows selecting support for the
  // KHR_lights_punctual extension.
  if (m_punctual) {
    // point/directional/spot
    extensions |= fastgltf::Extensions::KHR_lights_punctual;
  }

  fastgltf::Parser parser(extensions);

  constexpr auto gltfOptions =
    fastgltf::Options::None | fastgltf::Options::DontRequireValidAssetMember |
    fastgltf::Options::LoadExternalBuffers | fastgltf::Options::DecomposeNodeMatrices |
    fastgltf::Options::LoadExternalImages;

  fastgltf::GltfFileStream data(path);

  const auto type = fastgltf::determineGltfFileType(data);

  fastgltf::Expected<fastgltf::Asset> asset(fastgltf::Error::None);

  std::filesystem::path pathParent = path.parent_path();

  if (pathParent.empty()) {
    pathParent = std::filesystem::path("./");
  }

  if (type == fastgltf::GltfType::glTF) {
    asset = parser.loadGltf(data, pathParent, gltfOptions);
  } else if (type == fastgltf::GltfType::GLB) {
    asset = parser.loadGltfBinary(data, pathParent, gltfOptions);
  } else // if (type == Invalid)
  {
    std::cerr << "ERROR: determineGltfFileType returned Invalid\n";
    throw std::runtime_error("loadGLTF() Invalid file type");
  }

  if (asset.error() != fastgltf::Error::None) {
    std::cerr << "ERROR: loadGLTF() failed with error '" << fastgltf::getErrorMessage(asset.error())
              << "'\n";
    throw std::runtime_error("loadGLTF() Failed");
  }

  // Store in m_assets instead of m_asset
  m_assets.push_back(std::move(asset.get()));
}

fastgltf::Extensions Application::getGLTFExtensions() {
  return fastgltf::Extensions::KHR_materials_anisotropy |
         fastgltf::Extensions::KHR_materials_clearcoat |
         fastgltf::Extensions::KHR_materials_emissive_strength |
         fastgltf::Extensions::KHR_materials_ior | fastgltf::Extensions::KHR_materials_iridescence |
         fastgltf::Extensions::KHR_materials_sheen | fastgltf::Extensions::KHR_materials_specular |
         fastgltf::Extensions::KHR_materials_transmission |
         fastgltf::Extensions::KHR_materials_unlit | fastgltf::Extensions::KHR_materials_variants |
         fastgltf::Extensions::KHR_materials_volume | fastgltf::Extensions::KHR_mesh_quantization |
         fastgltf::Extensions::KHR_texture_transform |
         fastgltf::Extensions::EXT_meshopt_compression;
}

fastgltf::Asset Application::loadGLTFIntoAsset(const std::filesystem::path& path) {
  std::cout << "loadGLTFIntoAsset(" << path << ")\n";

  if (!std::filesystem::exists(path)) {
    std::cerr << "WARNING: loadGLTFIntoAsset() filename " << path << " not found.\n";
    throw std::runtime_error("loadGLTFIntoAsset() File not found");
  }

  fastgltf::Extensions extensions = getGLTFExtensions();

  // The command line parameter --punctual (-p) <int> allows selecting support for the
  // KHR_lights_punctual extension.
  if (m_punctual) {
    // point/directional/spot
    extensions |= fastgltf::Extensions::KHR_lights_punctual;
  }

  fastgltf::Parser parser(extensions);

  constexpr auto gltfOptions =
    fastgltf::Options::None | fastgltf::Options::DontRequireValidAssetMember |
    fastgltf::Options::LoadExternalBuffers | fastgltf::Options::DecomposeNodeMatrices |
    fastgltf::Options::LoadExternalImages;

  fastgltf::GltfFileStream data(path);

  const auto type = fastgltf::determineGltfFileType(data);

  fastgltf::Expected<fastgltf::Asset> asset(fastgltf::Error::None);

  std::filesystem::path pathParent = path.parent_path();

  if (pathParent.empty()) {
    pathParent = std::filesystem::path("./");
  }

  if (type == fastgltf::GltfType::glTF) {
    asset = parser.loadGltf(data, pathParent, gltfOptions);
  } else if (type == fastgltf::GltfType::GLB) {
    asset = parser.loadGltfBinary(data, pathParent, gltfOptions);
  } else // if (type == Invalid)
  {
    std::cerr << "ERROR: determineGltfFileType returned Invalid\n";
    throw std::runtime_error("loadGLTFIntoAsset() Invalid file type");
  }

  if (asset.error() != fastgltf::Error::None) {
    std::cerr << "ERROR: loadGLTFIntoAsset() failed with error '"
              << fastgltf::getErrorMessage(asset.error()) << "'\n";
    throw std::runtime_error("loadGLTFIntoAsset() Failed");
  }

  // Return the loaded asset
  std::cout << "Successfully loaded GLB with " << asset.get().meshes.size() << " meshes"
            << std::endl;
  return std::move(asset.get());
}

fastgltf::Asset Application::loadGLTFIntoAsset(const std::string& glb_path) {
  return loadGLTFIntoAsset(std::filesystem::path(glb_path));
}

size_t Application::calculateTotalNodes() {
  size_t totalNodes = 0;
  for (const auto& asset : m_assets) {
    totalNodes += asset.nodes.size();
  }
  return totalNodes;
}

void Application::initNodes() {
  if (m_assets.empty()) return;
  if (m_nodes.empty()) {
    m_nodes.reserve(calculateTotalNodes());
  }
  std::cout << calculateTotalNodes() << " node(s) to initialize across " << m_assets.size()
            << " assets" << std::endl;

  for (const auto& asset : m_assets) {
    for (const fastgltf::Node& gltf_node : asset.nodes) {
      dev::Node& node = m_nodes.emplace_back();
      
      // Set mesh index if available
      if (gltf_node.meshIndex.has_value()) {
        node.indexMesh = static_cast<int>(gltf_node.meshIndex.value());
      }

      // Process transform data (matrix and TRS are mutually exclusive)
      if (const auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&gltf_node.transform)) {
        node.matrix = glm::make_mat4x4(matrix->data());
        node.isDirtyMatrix = false;
      } else if (const auto* transform = std::get_if<fastgltf::TRS>(&gltf_node.transform)) {
        node.translation = glm::make_vec3(transform->translation.data());
        node.rotation = glm::quat::wxyz(transform->rotation[3], transform->rotation[0],
                                        transform->rotation[1], transform->rotation[2]);
        node.scale = glm::make_vec3(transform->scale.data());
        node.isDirtyMatrix = true;
      }
    }
  }
}

void Application::initImages() {
  std::cout << "\n=== Loading GLB Images ===" << std::endl;
  std::cout << "Current images count (from ground material): " << m_images.size() << std::endl;
  
  size_t assetIndex = 0;
  for (const fastgltf::Asset& asset : m_assets) {
    size_t startImageIndex = m_images.size();
    std::cout << "Loading " << asset.images.size() << " images from asset " << assetIndex 
              << ", starting at global index " << startImageIndex << std::endl;
    
    size_t localImageIdx = 0;
    for (const fastgltf::Image& image : asset.images) {
      std::visit(
        fastgltf::visitor{
          [](const auto& /* arg */) {},
          [&](const fastgltf::sources::URI& filePath) {
            MY_ASSERT(filePath.fileByteOffset == 0); // No offsets supported with stbi.
            MY_ASSERT(filePath.uri.isLocalPath());   // Loading only local files.
            int width;
            int height;
            int components;

            const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());

            unsigned char* data = stbi_load(path.c_str(), &width, &height, &components, 4);

            if (data != nullptr) {
              addImage(width, height, 8, 4, data);
              std::cout << "  Image " << localImageIdx << " (" << image.name << "): " 
                        << width << "x" << height << " -> global index " << (m_images.size()-1) << std::endl;
            } else {
              std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
              const unsigned char texel[4] = {0xFF, 0x00, 0xFF, 0xFF};
              addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
            }

            stbi_image_free(data);
          },

          [&](const fastgltf::sources::Array& vector) {
            int width;
            int height;
            int components;

            unsigned char* data = stbi_load_from_memory(
              reinterpret_cast<const stbi_uc*>(vector.bytes.data()),
              static_cast<int>(vector.bytes.size()), &width, &height, &components, 4);

            if (data != nullptr) {
              addImage(width, height, 8, 4, data);
              std::cout << "  Image " << localImageIdx << " (" << image.name << "): " 
                        << width << "x" << height << " (embedded) -> global index " << (m_images.size()-1) << std::endl;
            } else {
              std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
              const unsigned char texel[4] = {0xFF, 0x00, 0xFF, 0xFF};
              addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
            }

            stbi_image_free(data);
          },

          [&](const fastgltf::sources::BufferView& view) {
            const auto& bufferView = asset.bufferViews[view.bufferViewIndex];
            const auto& buffer = asset.buffers[bufferView.bufferIndex];

            std::visit(
              fastgltf::visitor{
                // We only care about Arrays here, because we specify LoadExternalBuffers, meaning
                // all buffers are already loaded into a vector.
                [](const auto& /* arg */) {},

                [&](const fastgltf::sources::Array& vector) {
                  int width;
                  int height;
                  int components;

                  unsigned char* data = stbi_load_from_memory(
                    reinterpret_cast<const stbi_uc*>(vector.bytes.data()) + bufferView.byteOffset,
                    static_cast<int>(bufferView.byteLength), &width, &height, &components, 4);

                  if (data != nullptr) {
                    addImage(width, height, 8, 4, data);
                  } else {
                    std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name
                              << '\n';
                    const unsigned char texel[4] = {0xFF, 0x00, 0xFF, 0xFF};
                    addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
                  }

                  stbi_image_free(data);
                }},
              buffer.data);
          },
        },
        image.data);
      localImageIdx++;
    }
    assetIndex++;
  }
  
  std::cout << "Total images loaded: " << m_images.size() << std::endl;
}

void Application::initTextures(size_t groundImageCount, size_t groundSamplerCount) {
  std::cout << "\n=== Loading GLB Textures ===" << std::endl;
  
  // Track image offset for each asset
  // IMPORTANT: Start from groundImageCount to account for ground material images
  size_t imageOffset = groundImageCount;
  size_t assetIndex = 0;
  
  std::cout << "Starting image offset (from ground material): " << groundImageCount << std::endl;
  
  for (const fastgltf::Asset& asset : m_assets) {
    std::cout << "Loading " << asset.textures.size() << " textures from asset " << assetIndex 
              << " (image offset: " << imageOffset << ")" << std::endl;
    
    std::vector<int> sRGB(asset.textures.size(), 0);

    for (const fastgltf::Material& material : asset.materials) {
      if (material.pbrData.baseColorTexture.has_value()) {
        const fastgltf::TextureInfo& textureInfo = material.pbrData.baseColorTexture.value();
        sRGB[textureInfo.textureIndex] = 1;
      }
      if (material.emissiveTexture.has_value()) {
        const fastgltf::TextureInfo& textureInfo = material.emissiveTexture.value();
        sRGB[textureInfo.textureIndex] = 1;
      }
      if (material.specular != nullptr && material.specular->specularColorTexture.has_value()) {
        const fastgltf::TextureInfo& textureInfo = material.specular->specularColorTexture.value();
        sRGB[textureInfo.textureIndex] = 1;
      }
      if (material.sheen != nullptr && material.sheen->sheenColorTexture.has_value()) {
        const fastgltf::TextureInfo& textureInfo = material.sheen->sheenColorTexture.value();
        sRGB[textureInfo.textureIndex] = 1;
      }
    }

    // Textures. These refer to previously loaded images.
    for (size_t i = 0; i < asset.textures.size(); ++i) {
      const fastgltf::Texture& texture = asset.textures[i];

      // Default to wrap repeat and linear filtering when there is no sampler.
      cudaTextureAddressMode address_s = cudaAddressModeWrap;
      cudaTextureAddressMode address_t = cudaAddressModeWrap;
      cudaTextureFilterMode filter = cudaFilterModeLinear;

      if (texture.samplerIndex.has_value()) {
        MY_ASSERT(texture.samplerIndex.value() < asset.samplers.size());
        const auto& sampler = asset.samplers[texture.samplerIndex.value()];

        address_s = utils::getTextureAddressMode(sampler.wrapS);
        address_t = utils::getTextureAddressMode(sampler.wrapT);

        if (sampler.minFilter.has_value()) {
          fastgltf::Filter minFilter = sampler.minFilter.value();

          switch (minFilter) {
            // This renderer is not downloading mipmaps.
            // Pick the filter depending on the 2D filtering which is the first.
          case fastgltf::Filter::Nearest:
          case fastgltf::Filter::NearestMipMapNearest:
          case fastgltf::Filter::NearestMipMapLinear:
            filter = cudaFilterModePoint;
            break;

          case fastgltf::Filter::Linear:
          case fastgltf::Filter::LinearMipMapNearest:
          case fastgltf::Filter::LinearMipMapLinear:
          default:
            filter = cudaFilterModeLinear;
            break;
          }
        }
      }

      MY_ASSERT(texture.imageIndex.has_value());
      
      // Map local image index to global image index
      size_t localImageIndex = texture.imageIndex.value();
      size_t globalImageIndex = imageOffset + localImageIndex;
      
      std::cout << "  Texture " << i << ": local image " << localImageIndex 
                << " -> global image " << globalImageIndex 
                << " (sRGB: " << sRGB[i] << ")" << std::endl;
      
      addSampler(address_s, address_t, filter, globalImageIndex, sRGB[i]);
    }
    
    // Update offset for next asset
    imageOffset += asset.images.size();
    assetIndex++;
  }
  
  std::cout << "Total samplers created: " << m_samplers.size() << std::endl;
}

void Application::initMaterials(size_t groundSamplerCount) {
  std::cout << "\n=== Loading GLB Materials ===" << std::endl;
  std::cout << "Current materials count (including ground): " << m_materials.size() << std::endl;
  
  // Record the starting index for GLB materials (after ground material)
  size_t startMaterialIndex = m_materials.size();
  size_t totalMaterialsAdded = 0;
  
  // Track sampler offset for each asset (samplers were created in initTextures)
  // IMPORTANT: Start from groundSamplerCount to skip ground material samplers
  size_t samplerOffset = groundSamplerCount;
  size_t assetIndex = 0;
  
  std::cout << "Starting sampler offset (from ground material): " << groundSamplerCount << std::endl;
  
  for (const fastgltf::Asset& asset : m_assets) {
    std::cout << "Loading " << asset.materials.size() << " materials from asset " << assetIndex 
              << " (sampler offset: " << samplerOffset << ")" << std::endl;
    
    for (size_t index = 0; index < asset.materials.size(); ++index) {
      const fastgltf::Material& material = asset.materials[index];

      MaterialData mtl;
      // Correct index calculation: base index + count of materials added so far
      mtl.index = static_cast<int>(startMaterialIndex + totalMaterialsAdded);
      mtl.doubleSided = material.doubleSided;
      
      std::cout << "  Material " << index << " (" << material.name << ")" << std::endl;
      std::cout << "    Local index: " << index << " -> Global index: " << mtl.index << std::endl;

      // Force all GLB materials to be opaque to prevent transparency issues
      mtl.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
      mtl.alphaCutoff = 0.0f;
      
      std::cout << "    Alpha mode: FORCED OPAQUE (was " << 
                (material.alphaMode == fastgltf::AlphaMode::Opaque ? "Opaque" :
                 material.alphaMode == fastgltf::AlphaMode::Mask ? "Mask" :
                 material.alphaMode == fastgltf::AlphaMode::Blend ? "Blend" : "Unknown") << ")" << std::endl;

      mtl.baseColorFactor =
        make_float4(material.pbrData.baseColorFactor[0], material.pbrData.baseColorFactor[1],
                    material.pbrData.baseColorFactor[2], material.pbrData.baseColorFactor[3]);
      
      // Keep original colors for proper albedo display
      mtl.baseColorFactor.w = 1.0f; // Force alpha to 1.0 (fully opaque)
      
      std::cout << "    Base color factor: (" << mtl.baseColorFactor.x << ", " 
                << mtl.baseColorFactor.y << ", " << mtl.baseColorFactor.z << ", " 
                << mtl.baseColorFactor.w << ") (original colors)" << std::endl;
      std::cout << "    Metallic: " << mtl.metallicFactor << ", Roughness: " 
                << mtl.roughnessFactor << " (original values)" << std::endl;
      
      if (material.pbrData.baseColorTexture.has_value()) {
        detail::parseTextureInfo(m_samplers, material.pbrData.baseColorTexture.value(),
                                 mtl.baseColorTexture, samplerOffset);
        std::cout << "    Has base color texture: sampler object = " << mtl.baseColorTexture.object << std::endl;
      }

      mtl.metallicFactor = material.pbrData.metallicFactor; // Keep original metallic value
      mtl.roughnessFactor = material.pbrData.roughnessFactor; // Keep original roughness value
      if (material.pbrData.metallicRoughnessTexture.has_value()) {
        detail::parseTextureInfo(m_samplers, material.pbrData.metallicRoughnessTexture.value(),
                                 mtl.metallicRoughnessTexture, samplerOffset);
      }

      if (material.normalTexture.has_value()) {
        const auto& normalTextureInfo = material.normalTexture.value();

        mtl.normalTextureScale = normalTextureInfo.scale;
        detail::parseTextureInfo(m_samplers, normalTextureInfo, mtl.normalTexture, samplerOffset);
      }

      // Ambient occlusion should not really be required with a global illumination renderer,
      // but many glTF models are very low-resolution geometry and details are baked into normal and
      // occlusion maps.
      if (material.occlusionTexture.has_value()) {
        const auto& occlusionTextureInfo = material.occlusionTexture.value();

        mtl.occlusionTextureStrength = occlusionTextureInfo.strength;
        detail::parseTextureInfo(m_samplers, occlusionTextureInfo, mtl.occlusionTexture, samplerOffset);
      }

      mtl.emissiveStrength = material.emissiveStrength; // KHR_materials_emissive_strength
      mtl.emissiveFactor = make_float3(material.emissiveFactor[0], material.emissiveFactor[1],
                                       material.emissiveFactor[2]);
      if (material.emissiveTexture.has_value()) {
        detail::parseTextureInfo(m_samplers, material.emissiveTexture.value(), mtl.emissiveTexture, samplerOffset);
      }
      mtl.flags = 0;
      mtl.ior = material.ior;
      if (material.specular != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_SPECULAR;
        mtl.specularFactor = material.specular->specularFactor;
        if (material.specular->specularTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.specular->specularTexture.value(),
                                   mtl.specularTexture, samplerOffset);
        }
        mtl.specularColorFactor = make_float3(material.specular->specularColorFactor[0],
                                              material.specular->specularColorFactor[1],
                                              material.specular->specularColorFactor[2]);
        if (material.specular->specularColorTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.specular->specularColorTexture.value(),
                                   mtl.specularColorTexture, samplerOffset);
        }
      }
      if (material.transmission != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_TRANSMISSION;

        mtl.transmissionFactor = material.transmission->transmissionFactor;
        if (material.transmission->transmissionTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.transmission->transmissionTexture.value(),
                                   mtl.transmissionTexture, samplerOffset);
        }
      }
      if (material.volume != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_VOLUME;
        mtl.thicknessFactor = material.volume->thicknessFactor;
        mtl.attenuationDistance = material.volume->attenuationDistance;
        mtl.attenuationColor =
          make_float3(material.volume->attenuationColor[0], material.volume->attenuationColor[1],
                      material.volume->attenuationColor[2]);
      }
      if (material.clearcoat != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_CLEARCOAT;
        mtl.clearcoatFactor = material.clearcoat->clearcoatFactor;
        if (material.clearcoat->clearcoatTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.clearcoat->clearcoatTexture.value(),
                                   mtl.clearcoatTexture, samplerOffset);
        }
        mtl.clearcoatRoughnessFactor = material.clearcoat->clearcoatRoughnessFactor;
        if (material.clearcoat->clearcoatRoughnessTexture.has_value()) {
          detail::parseTextureInfo(m_samplers,
                                   material.clearcoat->clearcoatRoughnessTexture.value(),
                                   mtl.clearcoatRoughnessTexture, samplerOffset);
        }
        if (material.clearcoat->clearcoatNormalTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.clearcoat->clearcoatNormalTexture.value(),
                                   mtl.clearcoatNormalTexture, samplerOffset);
          mtl.isClearcoatNormalBaseNormal = (mtl.clearcoatNormalTexture == mtl.normalTexture);
        }
      }
      if (material.sheen != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_SHEEN;
        mtl.sheenColorFactor =
          make_float3(material.sheen->sheenColorFactor[0], material.sheen->sheenColorFactor[1],
                      material.sheen->sheenColorFactor[2]);
        if (material.sheen->sheenColorTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.sheen->sheenColorTexture.value(),
                                   mtl.sheenColorTexture, samplerOffset);
        }
        mtl.sheenRoughnessFactor = material.sheen->sheenRoughnessFactor;
        if (material.sheen->sheenRoughnessTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.sheen->sheenRoughnessTexture.value(),
                                   mtl.sheenRoughnessTexture, samplerOffset);
        }
      }
      if (material.anisotropy != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_ANISOTROPY;

        mtl.anisotropyStrength = material.anisotropy->anisotropyStrength;
        mtl.anisotropyRotation = material.anisotropy->anisotropyRotation;
        if (material.anisotropy->anisotropyTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.anisotropy->anisotropyTexture.value(),
                                   mtl.anisotropyTexture, samplerOffset);
        }
      }
      if (material.iridescence != nullptr) {
        mtl.flags |= FLAG_KHR_MATERIALS_IRIDESCENCE;

        mtl.iridescenceFactor = material.iridescence->iridescenceFactor;
        if (material.iridescence->iridescenceTexture.has_value()) {
          detail::parseTextureInfo(m_samplers, material.iridescence->iridescenceTexture.value(),
                                   mtl.iridescenceTexture, samplerOffset);
        }
        mtl.iridescenceIor = material.iridescence->iridescenceIor;
        mtl.iridescenceThicknessMinimum = material.iridescence->iridescenceThicknessMinimum;
        mtl.iridescenceThicknessMaximum = material.iridescence->iridescenceThicknessMaximum;
        if (material.iridescence->iridescenceThicknessTexture.has_value()) {
          detail::parseTextureInfo(m_samplers,
                                   material.iridescence->iridescenceThicknessTexture.value(),
                                   mtl.iridescenceThicknessTexture, samplerOffset);
        }
      }
      mtl.unlit = material.unlit;
      m_materialsOrg.push_back(mtl); // The original data inside the asset.
      m_materials.push_back(mtl);    // The materials changed by the GUI.
      
      totalMaterialsAdded++; // Increment counter for next material
    }
    
    // Update sampler offset for next asset
    samplerOffset += asset.textures.size();
    assetIndex++;
  }
  
  std::cout << "Total GLB materials loaded: " << totalMaterialsAdded << std::endl;
  std::cout << "Total materials in system: " << m_materials.size() << std::endl;
}

void Application::initMeshes() {
  uint32_t nPoints = 0;
  uint32_t nTris = 0;
  uint32_t nOthers = 0;
  uint32_t nPositions = 0;
  // ALL MESHES
  int dbgMeshId = -1;
  for (fastgltf::Asset& asset : m_assets) {
    for (const fastgltf::Mesh& gltf_mesh : asset.meshes) {
      dbgMeshId++;
      dev::HostMesh& hostMesh = m_hostMeshes.emplace_back();
      hostMesh.name = gltf_mesh.name;
      hostMesh.primitives.reserve(gltf_mesh.primitives.size());
      if (!gltf_mesh.weights.empty()) {
        hostMesh.weights.resize(gltf_mesh.weights.size());
        memcpy(hostMesh.weights.data(), gltf_mesh.weights.data(),
               gltf_mesh.weights.size() * sizeof(float));
      }
      int dbgPrimId = -1;
      for (const fastgltf::Primitive& primitive : gltf_mesh.primitives) {
        ++dbgPrimId;
        const dev::PrimitiveType primitiveType{detail::toDevPrimitiveType(primitive.type)};
        if (primitiveType == dev::PrimitiveType::Triangles)
          ++nTris;
        else if (primitiveType == dev::PrimitiveType::Points)
          ++nPoints;
        else if (primitiveType == dev::PrimitiveType::Undefined) {
          std::cerr << "glTF Primitive " << detail::getDevPrimitiveTypeName(primitive.type)
                    << " not yet implemented" << std::endl;
          ++nOthers;
          continue;
        } else {
          std::cerr << "ERROR Found unknown primitive type during initMeshes()" << std::endl;
          MY_ASSERT(false);
        }
        auto itPosition = primitive.findAttribute("POSITION");
        if (itPosition == primitive.attributes.end()) // Meshes MUST have a position attribute.
        {
          std::cerr << "ERROR: primitive has no POSITION attribute, skipped.\n";
          continue;
        }
        std::string name =
          "HostPrim_M" + std::to_string(dbgMeshId) + "_P" + std::to_string(dbgPrimId);
        dev::HostPrimitive& hostPrim = hostMesh.createNewPrimitive(primitiveType, name);
        int indexAccessor = static_cast<int>(itPosition->accessorIndex);
        nPositions +=
          utils::createHostBuffer("POSITION", asset, indexAccessor, fastgltf::AccessorType::Vec3,
                                  fastgltf::ComponentType::Float, 1.0f, hostPrim.positions);
        m_primitiveToHostBuffer[&primitive] = &hostPrim.positions;
        if (primitiveType != dev::PrimitiveType::Points) {
          indexAccessor = (primitive.indicesAccessor.has_value())
                            ? static_cast<int>(primitive.indicesAccessor.value())
                            : -1;
          utils::createHostBuffer("INDICES", asset, indexAccessor, fastgltf::AccessorType::Scalar,
                                  fastgltf::ComponentType::UnsignedInt, 0.0f, hostPrim.indices);
        }
        auto itNormal = primitive.findAttribute("NORMAL");
        indexAccessor =
          (itNormal != primitive.attributes.end()) ? static_cast<int>(itNormal->accessorIndex) : -1;
        const bool allowTangents = (0 <= indexAccessor);
        utils::createHostBuffer("NORMAL", asset, indexAccessor, fastgltf::AccessorType::Vec3,
                                fastgltf::ComponentType::Float, 0.0f, hostPrim.normals);
        auto itTangent = primitive.findAttribute("TANGENT");
        indexAccessor = (itTangent != primitive.attributes.end() && allowTangents)
                          ? static_cast<int>(itTangent->accessorIndex)
                          : -1;
        utils::createHostBuffer("TANGENT", asset, indexAccessor, fastgltf::AccessorType::Vec4,
                                fastgltf::ComponentType::Float, 1.0f, hostPrim.tangents);
        auto itColor = primitive.findAttribute("COLOR_0");
        indexAccessor =
          (itColor != primitive.attributes.end()) ? static_cast<int>(itColor->accessorIndex) : -1;
        utils::createHostBuffer("COLOR_0", asset, indexAccessor, fastgltf::AccessorType::Vec4,
                                fastgltf::ComponentType::Float, 1.0f,
                                hostPrim.colors); 
        for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j) {
          const std::string strTexcoord = std::string("TEXCOORD_") + std::to_string(j);
          auto itTexcoord = primitive.findAttribute(strTexcoord);
          indexAccessor = (itTexcoord != primitive.attributes.end())
                            ? static_cast<int>(itTexcoord->accessorIndex)
                            : -1;
          utils::createHostBuffer(strTexcoord.c_str(), asset, indexAccessor,
                                  fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float,
                                  0.0f, hostPrim.texcoords[j]);
        }
        for (int j = 0; j < NUM_ATTR_JOINTS; ++j) {
          std::string joints_str = std::string("JOINTS_") + std::to_string(j);
          auto itJoints = primitive.findAttribute(joints_str);
          indexAccessor = (itJoints != primitive.attributes.end())
                            ? static_cast<int>(itJoints->accessorIndex)
                            : -1;
          utils::createHostBuffer(joints_str.c_str(), asset, indexAccessor,
                                  fastgltf::AccessorType::Vec4,
                                  fastgltf::ComponentType::UnsignedShort, 0.0f, hostPrim.joints[j]);
        }

        for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j) {
          std::string weights_str = std::string("WEIGHTS_") + std::to_string(j);
          auto itWeights = primitive.findAttribute(weights_str);
          indexAccessor = (itWeights != primitive.attributes.end())
                            ? static_cast<int>(itWeights->accessorIndex)
                            : -1;
          utils::createHostBuffer(weights_str.c_str(), asset, indexAccessor,
                                  fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float,
                                  0.0f, hostPrim.weights[j]);
        }
        hostPrim.indexMaterial = (primitive.materialIndex.has_value())
                                   ? static_cast<int32_t>(primitive.materialIndex.value())
                                   : -1;
        for (size_t i = 0; i < primitive.mappings.size(); ++i) {
          const int index = primitive.mappings[i].has_value()
                              ? static_cast<int>(primitive.mappings[i].value())
                              : hostPrim.indexMaterial;

          hostPrim.mappings.push_back(index);
        }

        // Derive the current material index.
        hostPrim.currentMaterial =
          (primitive.mappings.empty()) ? hostPrim.indexMaterial : hostPrim.mappings[m_indexVariant];
      } // for primitive
    }   // for gltf_mesh

    std::cout << " *** Meshes found                        " << asset.meshes.size() << "\n"
              << " *** Triangle primitives found           " << nTris << "\n"
              << " *** Point       \"       \"               " << nPoints << "\n"
              << " *** Other       \"       \"               " << nOthers << " (ignored)\n"
              << " *** POSITION-s allocated in host memory " << nPositions << std::endl;
  }
}

void Application::createDevicePrimitive(dev::DevicePrimitive& devicePrim,
                                        const dev::HostPrimitive& hostPrim, const int skin) {
  MY_ASSERT(dev::PrimitiveType::Undefined != hostPrim.getPrimitiveType());

  devicePrim.setPrimitiveType(hostPrim.getPrimitiveType());

  devicePrim.numTargets = static_cast<int>(hostPrim.numTargets);

  utils::createDeviceBuffer(devicePrim.indices, hostPrim.indices); // unsigned int

  // Device Buffers for the base attributes.
  utils::createDeviceBuffer(devicePrim.positions,
                            hostPrim.positions); // float3 (this is the only mandatory attribute!)
  utils::createDeviceBuffer(devicePrim.tangents,
                            hostPrim.tangents); // float4 (.w == 1.0 or -1.0 for the handedness)
  utils::createDeviceBuffer(devicePrim.normals, hostPrim.normals); // float3
  utils::createDeviceBuffer(devicePrim.colors, hostPrim.colors);   // float4

  for (int i = 0; i < NUM_ATTR_TEXCOORDS; ++i) {
    utils::createDeviceBuffer(devicePrim.texcoords[i], hostPrim.texcoords[i]); // float2
  }
  // These are required on the device for the native CUDA skinning kernels.
  for (int i = 0; i < NUM_ATTR_JOINTS; ++i) {
    utils::createDeviceBuffer(devicePrim.joints[i], hostPrim.joints[i]); // ushort4
  }
  for (int i = 0; i < NUM_ATTR_WEIGHTS; ++i) {
    utils::createDeviceBuffer(devicePrim.weights[i], hostPrim.weights[i]); // float4
  }


  // Create the destination buffers for the skinned attributes only if the device mesh is under a
  // node with skin index. Otherwise the d_ptr remain null.
  if (0 <= skin) {
    // This will only create (and fill) device buffers for the base attributes which are present
    // inside the primitive.
    utils::createDeviceBuffer(devicePrim.positionsSkinned, hostPrim.positions); // float3
    utils::createDeviceBuffer(devicePrim.tangentsSkinned,
                              hostPrim.tangents); // float4 (.w == 1.0 or -1.0 for the handedness)
    utils::createDeviceBuffer(devicePrim.normalsSkinned, hostPrim.normals); // float3
  }

  // Set the final position attribute pointer. The OptixBuildInput vertexBuffers needs a pointer to
  // that.
  devicePrim.vertexBuffer = devicePrim.getPositionsPtr();

  devicePrim.currentMaterial = hostPrim.currentMaterial;
}

void Application::createDeviceMesh(dev::DeviceMesh& deviceMesh, const dev::KeyTuple key) {
  // Get the host mesh index and create all required DeviceBuffers.
  const dev::HostMesh& hostMesh = m_hostMeshes[key.idxHostMesh];

  deviceMesh.key = key; // This is unique per DeviceMesh.

  deviceMesh.primitives.reserve(hostMesh.primitives.size());

  for (const dev::HostPrimitive& hostPrim : hostMesh.primitives) {
    dev::DevicePrimitive& devicePrim = deviceMesh.primitives.emplace_back();
    devicePrim.setName(hostPrim.getName());

    createDevicePrimitive(devicePrim, hostPrim, key.idxSkin);
  }
}

void Application::addImage(const int32_t width, const int32_t height,
                           const int32_t bitsPerComponent, const int32_t numComponents,
                           const void* data) {
  // Allocate CUDA array in device memory
  int32_t pitch;
  cudaChannelFormatDesc channel_desc;

  if (bitsPerComponent == 8) {
    pitch = width * numComponents * sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();
  } else if (bitsPerComponent == 16) {
    pitch = width * numComponents * sizeof(uint16_t);
    channel_desc = cudaCreateChannelDesc<ushort4>();
  } else {
    std::cerr << "ERROR: addImage() Unsupported bitsPerComponent " << bitsPerComponent << '\n';
    throw std::runtime_error("addImage() Unsupported bitsPerComponent");
  }

  cudaArray_t cuda_array = nullptr;

  CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
  CUDA_CHECK(
    cudaMemcpy2DToArray(cuda_array, 0, 0, data, pitch, pitch, height, cudaMemcpyHostToDevice));

  m_images.push_back(cuda_array);
}

void Application::addSampler(cudaTextureAddressMode address_s, cudaTextureAddressMode address_t,
                             cudaTextureFilterMode filter, const size_t image_idx, const int sRGB) {
  cudaResourceDesc resDesc = {};

  resDesc.resType = cudaResourceTypeArray;
  MY_ASSERT(image_idx < m_images.size())
  resDesc.res.array.array = m_images[image_idx];

  cudaTextureDesc texDesc = {};

  texDesc.addressMode[0] = address_s;
  texDesc.addressMode[1] = address_t;
  texDesc.filterMode = filter;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;
  texDesc.maxAnisotropy = 1;
  texDesc.maxMipmapLevelClamp = 0;
  texDesc.minMipmapLevelClamp = 0;
  texDesc.mipmapFilterMode = cudaFilterModePoint; // No mipmap filtering.
  texDesc.borderColor[0] =
    1.0f; // DEBUG Why is the Khronos glTF-Sample Viewer using white for the border color?
  texDesc.borderColor[1] = 1.0f;
  texDesc.borderColor[2] = 1.0f;
  texDesc.borderColor[3] = 1.0f;
  // glTF uses sRGB for baseColor, specularColor, sheenColor and emissive texture RGB values, all
  // other texture data is linear. TextureLinearInterpolationTest.gltf requires that the texture
  // engine interpolates with sRGB enabled. Doing sRGB adjustments with pow(rgb, 2.2) inside the
  // shader is not producing the correct result because that is after linear texture interpolation.
  texDesc.sRGB = sRGB;

  // Create texture object.
  cudaTextureObject_t cuda_tex = 0;

  CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr));

  m_samplers.push_back(cuda_tex);
}

void Application::cleanup() {
  // Host and device asset cleanup.
  m_instances.clear();
  m_hostMeshes.clear();
  m_mapKeyTupleToDeviceMeshIndex.clear();
  m_deviceMeshes.clear();
  m_materialsOrg.clear();
  m_materials.clear();
  for (cudaTextureObject_t& sampler : m_samplers) {
    CUDA_CHECK(cudaDestroyTextureObject(sampler));
  }
  m_samplers.clear();
  for (cudaArray_t& image : m_images) {
    CUDA_CHECK(cudaFreeArray(image));
  }
  m_images.clear();
  m_nodes.clear();

  // Denoiser cleanup.
  cleanupDenoiser();

  // OptiX cleanup.
  if (m_pipeline) {
    OPTIX_CHECK(m_api.optixPipelineDestroy(m_pipeline));
    m_pipeline = 0;
  }

  for (OptixProgramGroup programGroup : m_programGroups) {
    OPTIX_CHECK(m_api.optixProgramGroupDestroy(programGroup));
  }
  m_programGroups.clear();

  for (OptixModule m : m_modules) {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
  m_modules.clear();

  if (m_optixContext) {
    OPTIX_CHECK(m_api.optixDeviceContextDestroy(m_optixContext));
    m_optixContext = 0;
  }

  // CUDA cleanup.
  if (m_cudaGraphicsResource != nullptr) {
    CU_CHECK(cuGraphicsUnregisterResource(m_cudaGraphicsResource));
  }

  if (m_sbt.raygenRecord) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    m_sbt.raygenRecord = 0;
  }
  if (m_sbt.missRecordBase) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    m_sbt.missRecordBase = 0;
  }
  if (m_sbt.hitgroupRecordBase) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
    m_sbt.hitgroupRecordBase = 0;
  }

  if (m_launchParameters.bufferAccum != 0 &&
      m_interop != INTEROP_PBO) // For INTEROP_PBO: bufferAccum contains the last PBO mapping, do
                                // not call cudaFree() on that.
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)));
  }

  if (m_launchParameters.bufferPicking != 0) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferPicking)));
  }

  m_growIas.clear();
  m_growIasTemp.clear();
  m_growInstances.clear();

  if (m_d_iasAABB != 0) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_iasAABB)));
    m_d_iasAABB = 0;
  }

  if (m_d_lightDefinitions != 0) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_lightDefinitions)));
    m_d_lightDefinitions = 0;
  }

  // OpenGL cleanup:
  if (m_pbo != 0) {
    glDeleteBuffers(1, &m_pbo);
  }
  if (m_hdrTexture != 0) {
    glDeleteTextures(1, &m_hdrTexture);
  }
  if (m_vboAttributes != 0) {
    glDeleteBuffers(1, &m_vboAttributes);
  }
  if (m_vboIndices != 0) {
    glDeleteBuffers(1, &m_vboIndices);
  }
  if (m_glslProgram != 0) {
    glDeleteProgram(m_glslProgram);
  }

  // Host side allocations.
  if (m_picSheenLUT != nullptr) {
    delete m_picSheenLUT;
  }
  if (m_texSheenLUT != nullptr) {
    delete m_texSheenLUT;
  }
  if (m_picEnv != nullptr) {
    delete m_picEnv;
  }
  if (m_texEnv != nullptr) {
    delete m_texEnv;
  }
}

void Application::buildDeviceMeshAccel(const int indexDeviceMesh, const bool rebuild) {
  // Build input flags depending on the different material configuration assigned to the individual
  // Primitive. Each alphaMode has a different anyhit program handling! Each element 0 has face
  // culling enabled, and element 1 has face culling disabled.
  //
  // Note that face-culling isn't really compatible with global illumination algorithms!
  // Materials which are fully transparent on one side and fully opaque on the other aren't
  // physically plausible. Neither reflections nor shadows will look as expected in some test scenes
  // (like NegativeScaleTest.gltf) which are explicitly built for local lighting in rasterizers.

  // ALPHA_MODE_OPAQUE materials do not need to call into anyhit programs!
  static const unsigned int inputFlagsOpaque[2] = {
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  // ALPHA_MODE_MASK materials are either fully opaque or fully transparent which is tested
  // inside the anyhit program by comparing the opacity against the alphaCutoff value.
  static const unsigned int inputFlagsMask[2] = {OPTIX_GEOMETRY_FLAG_NONE,
                                                 OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  // ALPHA_MODE_BLEND materials are using a stochastic opacity threshold which must be evaluated
  // only once per primitive.
  static const unsigned int inputFlagsBlend[2] = {
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL |
      OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  dev::DeviceMesh& deviceMesh = m_deviceMeshes[indexDeviceMesh];

  if (!deviceMesh.isDirty) {
    MY_ASSERT(deviceMesh.gas != 0 && deviceMesh.d_gas != 0);
    return; // Nothing to do for this device mesh.
  }

  // The device mesh key is unique, so it can be used to determine if a DeviceMesh is morphed or
  // skinned during animation.
  const bool compact = !(0 <= deviceMesh.key.idxNode || 0 <= deviceMesh.key.idxSkin);

  OptixAccelBuildOptions accelBuildOptions = {};

  // When the AS should be compacted, that means it's not changing dynamically with morphing and/or
  // skinning animations.
  if (compact) {
    // Non-animated vertex attributes are built to render as fast as possible.
    accelBuildOptions.buildFlags =
      OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    // accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // Always rebuild when reaching this.
  } else {
    // Meshes with animated vertex attributes (during node graph traversal) are built as fast as
    // possible and allow updates.
    accelBuildOptions.buildFlags =
      OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    // accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelBuildOptions.operation =
      (rebuild) ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;
  }

  // This builds one GAS per DeviceMesh but with build input and SBT hit record per DevicePrimitive
  // (with Triangles mode) to be able to use different input flags and material indices.

  MY_ASSERT(m_sceneExtent.isValid());

  std::vector<OptixBuildInput> buildInputs;
  buildInputs.reserve(deviceMesh.primitives.size());

  const auto sceneSize = m_sceneExtent.getDiameter();
  const float allSpheresRadius = m_sphereRadiusFraction * sceneSize;

  for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives) {
    OptixBuildInput buildInput = {};
    // * Set the build input for triangles, points (as OptiX spheres), ...
    // * Set material properties based on the primitive's current material.
    if (devicePrim.setupBuildInput(buildInput, // OUT
                                   m_materials, inputFlagsOpaque, inputFlagsMask, inputFlagsBlend,
                                   allSpheresRadius)) {
      buildInputs.push_back(buildInput);
      /*if (buildInput.type == OPTIX_BUILD_INPUT_TYPE_SPHERES)
      std::cout << "allSpheresRadius " << allSpheresRadius << std::endl;
      */
    }
  } // for deviceMesh.primitives

  if (!buildInputs.empty()) {
    // If this routine is called with rebuild more than once for a mesh, free the d_gas of this mesh
    // and rebuild it.
    if (rebuild && deviceMesh.d_gas) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(deviceMesh.d_gas)));

      deviceMesh.d_gas = 0;
      deviceMesh.gas = 0;
    }

    OptixAccelBufferSizes accelBufferSizes = {};

    OPTIX_CHECK(m_api.optixAccelComputeMemoryUsage(
      m_optixContext, &accelBuildOptions, buildInputs.data(),
      static_cast<unsigned int>(buildInputs.size()), &accelBufferSizes));

    if (compact) // This is always a build operation.
    {
      CUdeviceptr d_gas; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas), accelBufferSizes.outputSizeInBytes));

      CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), accelBufferSizes.tempSizeInBytes));

      OptixAccelEmitDesc accelEmit = {};

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&accelEmit.result),
                            8)); // Room for size_t for the compacted size.
      accelEmit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

      OPTIX_CHECK(m_api.optixAccelBuild(
        m_optixContext, m_cudaStream, &accelBuildOptions, buildInputs.data(),
        static_cast<unsigned int>(buildInputs.size()), d_temp, accelBufferSizes.tempSizeInBytes,
        d_gas, accelBufferSizes.outputSizeInBytes, &deviceMesh.gas,
        &accelEmit, // Emitted property: compacted size
        1));        // Number of emitted properties.

      size_t sizeCompact;

      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(&sizeCompact), (const void*)accelEmit.result,
                            sizeof(size_t), cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(accelEmit.result)));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));

      // Compact the AS only when possible. This can save more than half the memory on RTX boards.
      if (sizeCompact < accelBufferSizes.outputSizeInBytes) {
        CUdeviceptr d_gasCompact;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasCompact), sizeCompact));

        OPTIX_CHECK(m_api.optixAccelCompact(m_optixContext, m_cudaStream, deviceMesh.gas,
                                            d_gasCompact, sizeCompact, &deviceMesh.gas));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas)));

        deviceMesh.d_gas = d_gasCompact;
      } else {
        deviceMesh.d_gas = d_gas;
      }
    } else // if (!compact) // Means morphing or skinning animation. This can be an initial build or
           // an update operation.
    {
      size_t sizeTemp =
        accelBufferSizes
          .tempUpdateSizeInBytes; // Temporary memory required for an update operation.

      if (rebuild) {
        sizeTemp =
          accelBufferSizes.tempSizeInBytes; // Temporary memory required for a full build operation.

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMesh.d_gas),
                              accelBufferSizes.outputSizeInBytes)); // d_gas has been freed above.
      }

      CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), sizeTemp));

      OPTIX_CHECK(m_api.optixAccelBuild(
        m_optixContext, m_cudaStream, &accelBuildOptions, buildInputs.data(),
        static_cast<unsigned int>(buildInputs.size()), d_temp, sizeTemp, deviceMesh.d_gas,
        accelBufferSizes.outputSizeInBytes, &deviceMesh.gas, nullptr, 0));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));
    }
  }

  deviceMesh.isDirty = false;
}

// This is called when changing materials.
void Application::buildDeviceMeshAccels(const bool rebuild) {
  for (int indexDeviceMesh = 0; indexDeviceMesh < static_cast<int>(m_deviceMeshes.size());
       ++indexDeviceMesh) {
    buildDeviceMeshAccel(indexDeviceMesh, rebuild);
  }
}

void Application::buildInstanceAccel(const bool rebuild) {
  const size_t numInstances = m_instances.size();
  if (numInstances == 0) 
  {
    m_sceneExtent.toUnity();
    m_ias = 0;
    return;
  }
  m_sceneExtent.toInvalid();
  if (m_d_iasAABB == 0) 
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_iasAABB), 6 * sizeof(float)));
  }

  // Filter instances based on visibility
  std::vector<OptixInstance> optix_instances;
  std::vector<size_t> visibleInstanceIndices;
  
  unsigned int sbt_offset = 0;
  size_t dbgNumPrims = 0;

  // Debug: Print instance filtering info
  std::cout << "Instance filtering: m_assets.size()=" << m_assets.size() 
            << ", m_assetVisibility.size()=" << m_assetVisibility.size() 
            << ", m_instances.size()=" << m_instances.size() << std::endl;
  
  for (size_t i = 0; i < m_instances.size(); ++i) {
    const dev::Instance& instance = m_instances[i];
    
    // Check if this instance should be visible
    bool isVisible = true;
    
    // Ground instance (index 0) is always visible
    if (i > 0) {
      // Check if this is a GLB asset instance
      // GLB assets are at indices 1 to m_assets.size()
      // Procedural primitives are at indices m_assets.size()+1 and beyond
      if (i <= m_assets.size()) {
        // This is a GLB asset instance
        size_t assetIndex = i - 1; // Convert instance index to asset index
        if (assetIndex < m_assetVisibility.size()) {
          isVisible = m_assetVisibility[assetIndex];
          std::cout << "Instance " << i << " (GLB asset " << assetIndex << "): " 
                    << (isVisible ? "visible" : "hidden") << std::endl;
        }
      } else {
        std::cout << "Instance " << i << " (procedural primitive): always visible" << std::endl;
      }
    } else {
      std::cout << "Instance " << i << " (ground): always visible" << std::endl;
    }
    
    if (isVisible) {
      OptixInstance optix_instance;
      memset(&optix_instance, 0, sizeof(OptixInstance));

      optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
      optix_instance.instanceId = static_cast<unsigned int>(i); // Keep original instance ID
      optix_instance.sbtOffset = sbt_offset;
      
      // Set visibility mask based on asset visibility
      if (isVisible) {
        optix_instance.visibilityMask = m_visibilityMask; // Fully visible
      } else {
        optix_instance.visibilityMask = 0; // Completely hidden
      }
      
      optix_instance.traversableHandle = m_deviceMeshes[instance.indexDeviceMesh].gas;

      utils::setInstanceTransform(optix_instance, instance.transform);
      
      optix_instances.push_back(optix_instance);
      visibleInstanceIndices.push_back(i);
      
      sbt_offset += static_cast<unsigned int>(m_deviceMeshes[instance.indexDeviceMesh].primitives.size()) * NUM_RAY_TYPES;
      dbgNumPrims += m_deviceMeshes[instance.indexDeviceMesh].primitives.size();
    } else {
      // For hidden instances, we still need to account for their SBT space
      // This ensures SBT offsets remain consistent
      sbt_offset += static_cast<unsigned int>(m_deviceMeshes[instance.indexDeviceMesh].primitives.size()) * NUM_RAY_TYPES;
    }
  }
  
  const size_t numVisibleInstances = optix_instances.size();

  const size_t sizeBytesInstances = sizeof(OptixInstance) * numVisibleInstances;

  m_growInstances.grow(sizeBytesInstances);

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_growInstances.d_ptr), optix_instances.data(),
                        sizeBytesInstances, cudaMemcpyHostToDevice));

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

  buildInput.instanceArray.instances = m_growInstances.d_ptr;
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(numVisibleInstances);

  OptixAccelBuildOptions accelBuildOptions = {};

  // The IAS can always be updated for animations or on some material parameter changes (alphaMode,
  // doubleSided, volume).
  accelBuildOptions.buildFlags = getBuildFlags();

  accelBuildOptions.operation =
    (rebuild) ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

  OptixAccelBufferSizes accelBufferSizes = {};

  OPTIX_CHECK(m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1,
                                                 &accelBufferSizes));

  // This grow() assumes outputSizeInBytes for an update operation is always less or equal to the
  // previous build operation.
  MY_ASSERT(rebuild || (!rebuild && accelBufferSizes.outputSizeInBytes <= m_growIas.size));
  m_growIas.grow(accelBufferSizes.outputSizeInBytes);

  const size_t tempSizeInBytes =
    (rebuild) ? accelBufferSizes.tempSizeInBytes : accelBufferSizes.tempUpdateSizeInBytes;

  m_growIasTemp.grow(tempSizeInBytes);

  OptixAccelEmitDesc emitDesc = {};

  // Emit the top-level AABB to know the scene size.
  emitDesc.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitDesc.result = m_d_iasAABB;

  // TODO timer.start();

  OPTIX_CHECK(m_api.optixAccelBuild(m_optixContext, m_cudaStream, &accelBuildOptions, &buildInput,
                                    1, // num build inputs
                                    m_growIasTemp.d_ptr, m_growIasTemp.size, m_growIas.d_ptr,
                                    m_growIas.size, &m_ias, &emitDesc, 1));
  // TODO std::cout << "Build time " << timer.getElapsedSeconds() << std::endl;

  glm::vec3 sceneAABB[2];
  CUDA_CHECK(cudaMemcpy(&sceneAABB[0].x, reinterpret_cast<const void*>(emitDesc.result),
                        6 * sizeof(float), cudaMemcpyDeviceToHost));
  // NOTE: change the scene extent!
  m_sceneExtent.set(sceneAABB);
  MY_ASSERT(m_sceneExtent.isValid());
}

void Application::initPipeline() {
  // Set all module and pipeline options.

  // OptixModuleCompileOptions
  m_mco = {};

  m_mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if USE_DEBUG_EXCEPTIONS
  m_mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
  m_mco.debugLevel =
    OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Full debug. Never profile kernels with this setting!
#else
  m_mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
  m_mco.debugLevel =
    OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // PERF Must use OPTIX_COMPILE_DEBUG_LEVEL_MODERATE to
                                       // profile code with Nsight Compute!
#endif // USE_DEBUG_EXCEPTIONS

  // OptixPipelineCompileOptions
  m_pco = {};

  m_pco.usesMotionBlur = 0;
  m_pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pco.numPayloadValues = 2; // Need only two register for the payload pointer.
  m_pco.numAttributeValues =
    2; // For the two barycentric coordinates of built-in triangles. (The required minimum value.)
#ifndef NDEBUG // USE_DEBUG_EXCEPTIONS
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_USER;
#else
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  m_pco.pipelineLaunchParamsVariableName = "theLaunchParameters";

  // Supporting built-in tris and spheres at this time.
  m_pco.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
                                                           OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);

  // OptixPipelineLinkOptions
  m_plo = {};

  m_plo.maxTraceDepth = MAX_TRACE_DEPTH;

  // OptixProgramGroupOptions
  m_pgo = {}; // Just a placeholder.

  // Build the module path names.
  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create
  // modules. Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and
  // CUDA 11.7 or newer are found.

  const std::string path(CUDA_PROGRAMS_PATH);

#if defined(USE_OPTIX_IR)
  const std::string extension(".optixir");
#else
  const std::string extension(".ptx");
#endif

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  m_moduleFilenames[MODULE_ID_RAYGENERATION] = path + std::string("raygen") + extension;
  m_moduleFilenames[MODULE_ID_EXCEPTION] = path + std::string("exception") + extension;
  m_moduleFilenames[MODULE_ID_MISS] = path + std::string("miss") + extension;
  m_moduleFilenames[MODULE_ID_HIT] = path + std::string("hit") + extension;
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE] =
    path + std::string("light_sample") + extension; // Direct callable programs.

  // Create all modules.

  MY_ASSERT(NUM_RAY_TYPES == 2); // The following code only works for two raytypes.

  m_modules.resize(NUM_MODULE_IDENTIFIERS);

  for (size_t i = 0; i < m_moduleFilenames.size(); ++i) {
    std::vector<char> programData = readData(m_moduleFilenames[i]);

    OPTIX_CHECK(m_api.optixModuleCreate(m_optixContext, &m_mco, &m_pco, programData.data(),
                                        programData.size(), nullptr, nullptr, &m_modules[i]));
  }

  // For spheres we need this:
  OptixBuiltinISOptions builtin_is_options = {};

  builtin_is_options.usesMotionBlur = m_pco.usesMotionBlur;
  builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
  builtin_is_options.buildFlags = getBuildFlags();
  OPTIX_CHECK(m_api.optixBuiltinISModuleGet(m_optixContext, &m_mco, &m_pco, &builtin_is_options,
                                            &m_moduleBuiltinISSphere));

  // TODO the same for curves.

  // Create the program groups descriptions.

  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_GROUP_IDS);
  memset(programGroupDescriptions.data(), 0,
         sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());

  OptixProgramGroupDesc* pgd = &programGroupDescriptions[PGID_RAYGENERATION];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module = m_modules[MODULE_ID_RAYGENERATION];

  if (m_interop != INTEROP_IMG) {
    pgd->raygen.entryFunctionName = "__raygen__path_tracer";
  } else {
    pgd->raygen.entryFunctionName = "__raygen__path_tracer_surface";
  }

  pgd = &programGroupDescriptions[PGID_EXCEPTION];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->exception.module = m_modules[MODULE_ID_EXCEPTION];
  pgd->exception.entryFunctionName = "__exception__all";

  pgd = &programGroupDescriptions[PGID_MISS_RADIANCE];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = m_modules[MODULE_ID_MISS];

  switch (m_missID) {
  case 0:
  default: // Every other ID means there is no environment light.
    pgd->miss.entryFunctionName = "__miss__env_null";
    break;
  case 1:
    pgd->miss.entryFunctionName = "__miss__env_constant";
    break;
  case 2:
    pgd->miss.entryFunctionName = "__miss__env_sphere";
    break;
  }

  pgd = &programGroupDescriptions[PGID_MISS_SHADOW];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = m_modules[MODULE_ID_MISS];
  pgd->miss.entryFunctionName = "__miss__shadow"; // alphaMode OPAQUE is not using anyhit or closest
                                                  // hit programs for the shadow ray.

  // TRIANGLES: The hit records for the radiance ray.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_TRIANGLES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH = m_modules[MODULE_ID_HIT];
  //  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  //  TRIANGLES: The hit records for the shadow ray
  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_TRIANGLES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  //  SPHERES: The hit records for the radiance ray.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_SPHERES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance_sphere";
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_sphere";
  pgd->hitgroup.moduleIS = m_moduleBuiltinISSphere;
  pgd->hitgroup.entryFunctionNameIS = 0;

  // SPHERES: The hit records for the shadow ray.
  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_SPHERES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_sphere";

  // TODO the same for curves

  // Light Sampler
  // Only one of the environment callables will ever be used, but both are required
  // for the proper direct callable index calculation for BXDFs using NUM_LIGHT_TYPES.
  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_CONSTANT];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_constant";

  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_SPHERE];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";

  pgd = &programGroupDescriptions[PGID_LIGHT_POINT];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_point";

  pgd = &programGroupDescriptions[PGID_LIGHT_SPOT];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_spot";

  pgd = &programGroupDescriptions[PGID_LIGHT_DIRECTIONAL];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_directional";

  // Create the program groups.

  m_programGroups.resize(programGroupDescriptions.size());

  OPTIX_CHECK(m_api.optixProgramGroupCreate(m_optixContext, programGroupDescriptions.data(),
                                            (unsigned int)programGroupDescriptions.size(), &m_pgo,
                                            nullptr, nullptr, m_programGroups.data()));

  // 3.) Create the pipeline.

  OPTIX_CHECK(m_api.optixPipelineCreate(m_optixContext, &m_pco, &m_plo, m_programGroups.data(),
                                        (unsigned int)m_programGroups.size(), nullptr, nullptr,
                                        &m_pipeline));

  // 4.) Calculate the stack size.
  // This is is always recommended and strictly required when using any direct or continuation
  // callables.
  OptixStackSizes ssp = {}; // Whole pipeline.

  for (OptixProgramGroup pg : m_programGroups) {
    OptixStackSizes ss;

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK(m_api.optixProgramGroupGetStackSize(pg, &ss, m_pipeline));
#else
    OPTIX_CHECK(m_api.optixProgramGroupGetStackSize(pg, &ss));
#endif

    ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
    ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
    ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
    ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
    ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
    ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
    ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
  }

  // Temporaries
  unsigned int cssCCTree =
    ssp.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
  unsigned int cssCHOrMSPlusCCTree = std::max(ssp.cssCH, ssp.cssMS) + cssCCTree;

  // Arguments
  unsigned int directCallableStackSizeFromTraversal =
    ssp.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible
               // stack size optimizations.
  unsigned int directCallableStackSizeFromState =
    ssp.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible
               // stack size optimizations.
  unsigned int continuationStackSize =
    ssp.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, m_plo.maxTraceDepth) - 1u) +
    std::min(1u, m_plo.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
  unsigned int maxTraversableGraphDepth = 2;

  OPTIX_CHECK(m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal,
                                              directCallableStackSizeFromState,
                                              continuationStackSize, maxTraversableGraphDepth));
}

void Application::initSBT() {
  {
    //
    // SBT RAYGEN
    //
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), sizeof(dev::EmptyRecord)));

    dev::EmptyRecord rg_sbt;

    OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_RAYGENERATION], &rg_sbt));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.raygenRecord), &rg_sbt,
                          sizeof(dev::EmptyRecord), cudaMemcpyHostToDevice));
  }

  {
    //
    // SBT MISS
    //
    const size_t miss_record_size = sizeof(dev::EmptyRecord);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.missRecordBase),
                          miss_record_size * NUM_RAY_TYPES));

    dev::EmptyRecord ms_sbt[NUM_RAY_TYPES];

    OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_RADIANCE], &ms_sbt[0]));
    OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_SHADOW], &ms_sbt[1]));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.missRecordBase), ms_sbt,
                          miss_record_size * NUM_RAY_TYPES, cudaMemcpyHostToDevice));

    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = NUM_RAY_TYPES;
  }

  {
    //
    // SBT HITGROUPS {RadianceRay, ShadowRay} x {Triangles, Spheres}
    //

    std::vector<dev::HitGroupRecord> hitGroupRecords;

    // Track which asset each instance belongs to
    // Instance 0 is ground, assets start from instance 1
    size_t instanceIndex = 0;
    for (const dev::Instance& instance : m_instances) {
      int assetIndex = -1; // Ground has assetIndex -1
      
      // Calculate asset index: instance 0 is ground, instance 1+ are assets
      if (instanceIndex > 0 && (instanceIndex - 1) < m_assets.size()) {
        assetIndex = static_cast<int>(instanceIndex - 1);
      }
      
      // std::cout << "Instance " << instance.indexDeviceMesh << std::endl;

      const dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];

      for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives) {
        // Geometry and material.
        dev::HitGroupRecord rec = {};

        // std::cout << "\tDevice Primitive " << devicePrim.name << ", type " <<
        // devicePrim.getPrimitiveType() << std::endl;

        if (devicePrim.getPrimitiveType() == dev::PrimitiveType::Triangles) {
          OPTIX_CHECK(
            m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_TRIANGLES], &rec));
          // std::cout << "[TRIS] optixSbtRecordPackHeader PGID_HIT_RADIANCE_TRIANGLES" <<
          // std::endl;

          GeometryData::TriangleMesh triangleMesh = {};

          // Indices
          triangleMesh.indices = reinterpret_cast<uint3*>(devicePrim.indices.d_ptr);
          // Attributes
          triangleMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
          triangleMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
          for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j) {
            triangleMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
          }
          triangleMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
          triangleMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
          for (int j = 0; j < NUM_ATTR_JOINTS; ++j) {
            triangleMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
          }
          for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j) {
            triangleMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
          }
          // triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.

          // Note that both trimesh and spheremesh have { positions, normals, colors }
          rec.data.geometryData.setTriangleMesh(triangleMesh);

          if (0 <= devicePrim.currentMaterial && devicePrim.currentMaterial < m_materials.size()) {
            rec.data.materialData = m_materials[devicePrim.currentMaterial];
          } else {
            std::cerr << "WARNING: Invalid material index " << devicePrim.currentMaterial
                      << " for primitive " << devicePrim.name
                      << " (materials count: " << m_materials.size() << ")" << std::endl;
            rec.data.materialData = MaterialData(); // These default materials cannot be edited!
          }
          
          // Set asset index for mask generation
          rec.data.assetIndex = assetIndex;

          hitGroupRecords.push_back(rec); // radiance ray - triangles

          OPTIX_CHECK(
            m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_TRIANGLES], &rec));
          // std::cout << "[TRIS] optixSbtRecordPackHeader PGID_HIT_SHADOW_TRIANGLES" << std::endl;
          
          // Set asset index for shadow ray too (same as radiance ray)
          rec.data.assetIndex = assetIndex;

          hitGroupRecords.push_back(rec); // shadow ray - triangles
        } else if (devicePrim.getPrimitiveType() == dev::PrimitiveType::Points) {
          OPTIX_CHECK(
            m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_SPHERES], &rec));
          // OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_TRIANGLES],
          // &rec)); std::cout << "[POINTS] optixSbtRecordPackHeader PGID_HIT_RADIANCE_SPHERES" <<
          // std::endl;

          GeometryData::SphereMesh pointMesh = {};

          // Attributes
          pointMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
          pointMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
          for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j) {
            // pointMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
          }
          pointMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
          // pointMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
          /*for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
          {
          pointMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
          }*/
          for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j) {
            // pointMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
          }
          // triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.

          // Note that both trimesh and spheremesh have { positions, normals, colors }
          rec.data.geometryData.setSphereMesh(pointMesh);
          
          // Set asset index for mask generation (spheres)
          rec.data.assetIndex = assetIndex;

          if (0 <= devicePrim.currentMaterial) {
            rec.data.materialData = m_materials[devicePrim.currentMaterial];
          } else {
            rec.data.materialData = MaterialData(); // These default materials cannot be edited!
          }

          hitGroupRecords.push_back(rec); // radiance ray - spheres

          OPTIX_CHECK(
            m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_SPHERES], &rec));
          // OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_TRIANGLES],
          // &rec)); std::cout << "[POINTS] optixSbtRecordPackHeader PGID_HIT_SHADOW_SPHERES" <<
          // std::endl;
          
          // Set asset index for shadow ray too (same as radiance ray)
          rec.data.assetIndex = assetIndex;

          hitGroupRecords.push_back(rec); // shadow ray - spheres
        } else {
          // ignore the primitive
          MY_ASSERT(false);
        }
      } // all primitives
      
      ++instanceIndex; // Move to next instance
    }   // all instances

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
                          hitGroupRecords.size() * sizeof(dev::HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitGroupRecords.data(),
                          hitGroupRecords.size() * sizeof(dev::HitGroupRecord),
                          cudaMemcpyHostToDevice));

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(dev::HitGroupRecord));
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(hitGroupRecords.size());
  }

  {
    //
    // SBT CALLABLES
    //
    const size_t call_record_size = sizeof(dev::EmptyRecord);

    const int numCallables = NUM_PROGRAM_GROUP_IDS - PGID_LIGHT_ENV_CONSTANT;
    MY_ASSERT(numCallables == 5);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase),
                          call_record_size * numCallables));

    dev::EmptyRecord call_sbt[numCallables];

    OPTIX_CHECK(
      m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_CONSTANT], &call_sbt[0]));
    OPTIX_CHECK(
      m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_SPHERE], &call_sbt[1]));
    OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_POINT], &call_sbt[2]));
    OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_SPOT], &call_sbt[3]));
    OPTIX_CHECK(
      m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_DIRECTIONAL], &call_sbt[4]));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), call_sbt,
                          call_record_size * numCallables, cudaMemcpyHostToDevice));

    m_sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(call_record_size);
    m_sbt.callablesRecordCount = numCallables;
  }
}

LightDefinition Application::createSphericalEnvironmentLight() {
  if (m_picEnv == nullptr) {
    m_picEnv = new Picture();
  }

  bool loadedEnv = false;
  std::string selectedHdrPath;
  
  // First try to use command line specified HDR file
  if (!m_pathEnv.empty()) {
    loadedEnv = m_picEnv->load(m_pathEnv, IMAGE_FLAG_2D);
    selectedHdrPath = m_pathEnv;
    std::cout << "Using command line specified HDR: " << m_pathEnv << std::endl;
  }
  
  // If no command line HDR or loading failed, select brightest HDR from HDRI directory for debug
  if (!loadedEnv) {
    std::cout << "\n=== Selecting Brightest HDR Environment (Debug Mode) ===" << std::endl;
    
    // Scan HDRI directory for .hdr files
    std::vector<std::string> availableHdrFiles;
    try {
      std::filesystem::path hdriDir(m_path_hdri);
      if (std::filesystem::exists(hdriDir) && std::filesystem::is_directory(hdriDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(hdriDir)) {
          if (entry.is_regular_file() && entry.path().extension() == ".hdr") {
            availableHdrFiles.push_back(entry.path().string());
          }
        }
      }
    } catch (const std::exception& e) {
      std::cout << "Error scanning HDRI directory: " << e.what() << std::endl;
    }
    
    if (!availableHdrFiles.empty()) {
      std::cout << "Found " << availableHdrFiles.size() << " HDR files in directory: " << m_path_hdri << std::endl;
      
      // Analyze directionality and brightness for better shadow generation
      std::vector<std::pair<std::string, std::pair<float, float>>> hdrScores; // {path, {directionality, brightness}}
    const float directionalityThreshold = 0.2f; // Moderate threshold for directional lighting
    const float brightnessThreshold = 0.4f; // Moderate threshold for bright enough
      std::cout << "Analyzing HDR directionality and brightness for shadow generation..." << std::endl;
      std::cout << "Directionality threshold: " << directionalityThreshold << ", Brightness threshold: " << brightnessThreshold << std::endl;
      
      int analyzedCount = 0;
      int aboveThresholdCount = 0;
      
      for (const auto& hdrPath : availableHdrFiles) {
        float directionality = analyzeHDRDirectionality(hdrPath);
        float brightness = analyzeHDRBrightness(hdrPath);
        analyzedCount++;
        
        if (directionality > 0.0f && brightness > 0.0f) {
          hdrScores.push_back({hdrPath, {directionality, brightness}});
          
          if (directionality >= directionalityThreshold && brightness >= brightnessThreshold) {
            aboveThresholdCount++;
            std::cout << "  " << std::filesystem::path(hdrPath).filename().string() 
                      << " - directionality: " << std::fixed << std::setprecision(3) << directionality 
                      << ", brightness: " << std::fixed << std::setprecision(3) << brightness << " ✓" << std::endl;
          } else {
            std::cout << "  " << std::filesystem::path(hdrPath).filename().string() 
                      << " - directionality: " << std::fixed << std::setprecision(3) << directionality 
                      << ", brightness: " << std::fixed << std::setprecision(3) << brightness << " (below threshold)" << std::endl;
          }
        } else {
          std::cout << "  " << std::filesystem::path(hdrPath).filename().string() 
                    << " - directionality: " << std::fixed << std::setprecision(3) << directionality 
                    << ", brightness: " << std::fixed << std::setprecision(3) << brightness << " (invalid)" << std::endl;
        }
        
        // Stop analyzing after finding enough directional HDRs or after checking reasonable number
        if (aboveThresholdCount >= 5 || analyzedCount >= 50) {
          std::cout << "Stopping analysis after " << analyzedCount << " files (found " << aboveThresholdCount << " above threshold, need 1)" << std::endl;
          break;
        }
      }
      
      std::cout << "Analysis complete: " << aboveThresholdCount << " HDR files above thresholds" << std::endl;
      
      // Sort by directionality (highest first) - prioritize directional lighting for shadows
      std::sort(hdrScores.begin(), hdrScores.end(), 
                [](const std::pair<std::string, std::pair<float, float>>& a, 
                   const std::pair<std::string, std::pair<float, float>>& b) {
                  return a.second.first > b.second.first; // Sort by directionality descending
                });
      
      // Debug: Print top 5 most directional HDRs
      std::cout << "\nTop 5 most directional HDR files:" << std::endl;
      int debugCount = std::min(5, (int)hdrScores.size());
      for (int i = 0; i < debugCount; ++i) {
        std::cout << "  " << (i+1) << ". " << std::filesystem::path(hdrScores[i].first).filename().string() 
                  << " - directionality: " << std::fixed << std::setprecision(3) << hdrScores[i].second.first 
                  << ", brightness: " << std::fixed << std::setprecision(3) << hdrScores[i].second.second << std::endl;
      }
      
      // Try to load the most directional HDR files until we find a valid one
      bool foundValidHdr = false;
      int attempts = 0;
      const int maxAttempts = std::min(10, (int)hdrScores.size()); // Try up to 10 most directional files
      
      while (!foundValidHdr && attempts < maxAttempts) {
        selectedHdrPath = hdrScores[attempts].first;
        attempts++;
        
        std::cout << "Testing most directional HDR " << attempts << ": " << selectedHdrPath 
                  << " (directionality: " << std::fixed << std::setprecision(3) << hdrScores[attempts-1].second.first 
                  << ", brightness: " << std::fixed << std::setprecision(3) << hdrScores[attempts-1].second.second << ")" << std::endl;
        
        // Try to load the selected HDR file
        loadedEnv = m_picEnv->load(selectedHdrPath, IMAGE_FLAG_2D);
        if (loadedEnv) {
          // Create a temporary texture to validate the HDR file
          Texture* tempTexture = new Texture(m_allocator);
          if (tempTexture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) {
            if (tempTexture->isValidHDR()) {
              std::cout << "✓ Successfully loaded and validated brightest HDR environment" << std::endl;
              foundValidHdr = true;
              delete tempTexture;
            } else {
              std::cout << "✗ HDR file contains invalid data (inf/NaN values), trying next brightest..." << std::endl;
              delete tempTexture;
              loadedEnv = false; // Reset for next attempt
            }
          } else {
            std::cout << "✗ Failed to create texture from HDR file, trying next brightest..." << std::endl;
            delete tempTexture;
            loadedEnv = false; // Reset for next attempt
          }
        } else {
          std::cout << "✗ Failed to load HDR file, trying next brightest..." << std::endl;
        }
      }
      
      if (!foundValidHdr) {
        std::cout << "⚠ No valid directional HDR files found, trying fallback to brightest HDR..." << std::endl;
        
        // Fallback: Sort by brightness and try the brightest ones
        std::sort(hdrScores.begin(), hdrScores.end(), 
                  [](const std::pair<std::string, std::pair<float, float>>& a, 
                     const std::pair<std::string, std::pair<float, float>>& b) {
                    return a.second.second > b.second.second; // Sort by brightness descending
                  });
        
        attempts = 0;
        const int maxFallbackAttempts = std::min(5, (int)hdrScores.size());
        
        while (!foundValidHdr && attempts < maxFallbackAttempts) {
          selectedHdrPath = hdrScores[attempts].first;
          attempts++;
          
          std::cout << "Fallback attempt " << attempts << ": " << selectedHdrPath 
                    << " (brightness: " << std::fixed << std::setprecision(3) << hdrScores[attempts-1].second.second << ")" << std::endl;
          
          loadedEnv = m_picEnv->load(selectedHdrPath, IMAGE_FLAG_2D);
          if (loadedEnv) {
            Texture* tempTexture = new Texture(m_allocator);
            if (tempTexture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) {
              if (tempTexture->isValidHDR()) {
                std::cout << "✓ Successfully loaded fallback HDR environment" << std::endl;
                foundValidHdr = true;
                delete tempTexture;
              } else {
                std::cout << "✗ HDR file contains invalid data, trying next brightest..." << std::endl;
                delete tempTexture;
        loadedEnv = false;
              }
            } else {
              std::cout << "✗ Failed to create texture from HDR file, trying next brightest..." << std::endl;
              delete tempTexture;
              loadedEnv = false;
            }
          } else {
            std::cout << "✗ Failed to load HDR file, trying next brightest..." << std::endl;
          }
        }
        
        if (!foundValidHdr) {
          std::cout << "⚠ No valid HDR files found even with fallback" << std::endl;
          loadedEnv = false;
        }
      }
    } else {
      std::cout << "No HDR files found in directory: " << m_path_hdri << std::endl;
    }
  }
  
  // Fallback to synthetic environment if no HDR file could be loaded
  if (!loadedEnv) {
    std::cout << "Falling back to synthetic environment lighting for better shadows" << std::endl;
    m_picEnv->generateEnvironmentSynthetic(1024, 512);
  }
  
  // Use only input HDR environment lighting (no synthetic lighting)
  // Removed forced synthetic environment to use only input HDR files

  // Create a new texture to keep the old texture intact in case anything goes wrong.
  Texture* texture = new Texture(m_allocator);

  if (!texture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) {
    delete texture;
    throw std::runtime_error("createSphericalEnvironmentLight() environment map creation failed");
  }

  if (m_texEnv != nullptr) {
    delete m_texEnv;
  }

  m_texEnv = texture;

  LightDefinition light = {}; // All unused fields are set to zero.

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Textured environment
  light.cdfU = m_texEnv->getCDF_U();
  light.cdfV = m_texEnv->getCDF_V();

  // Emisson texture. If not zero, scales emission.
  light.textureEmission = m_texEnv->getTextureObject();

  light.emission = make_float3(1.0f); // Standard intensity for HDR environment lighting

  light.typeLight = TYPE_LIGHT_ENV_SPHERE;

  light.area = 4.0f * M_PIf; // Unused.
  light.invIntegral = 1.0f / m_texEnv->getIntegral();
  

  // Emission texture width and height. Used to index the CDFs, see above.
  light.width = m_texEnv->getWidth();
  light.height = m_texEnv->getHeight();

  return light;
}

LightDefinition Application::createSphericalEnvironmentLightFromExistingTexture() {
  // Use existing m_texEnv without recreating it
  if (m_texEnv == nullptr) {
    throw std::runtime_error("createSphericalEnvironmentLightFromExistingTexture() called but m_texEnv is null");
  }

  LightDefinition light = {}; // All unused fields are set to zero.

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Textured environment
  light.cdfU = m_texEnv->getCDF_U();
  light.cdfV = m_texEnv->getCDF_V();

  // Emisson texture. If not zero, scales emission.
  light.textureEmission = m_texEnv->getTextureObject();

  light.emission = make_float3(1.0f); // Standard intensity for HDR environment lighting

  light.typeLight = TYPE_LIGHT_ENV_SPHERE;

  light.area = 4.0f * M_PIf; // Unused.
  light.invIntegral = 1.0f / m_texEnv->getIntegral();
  

  // Emission texture width and height. Used to index the CDFs, see above.
  light.width = m_texEnv->getWidth();
  light.height = m_texEnv->getHeight();

  return light;
}

bool Application::checkMaterialCompatibility(const MaterialData& material, float& recommendedBrightness) {
  std::cout << "\n=== Material Compatibility Check ===" << std::endl;
  
  // Analyze material properties
  float metallicFactor = material.metallicFactor;
  float roughnessFactor = material.roughnessFactor;
  float4 baseColor4 = material.baseColorFactor;
  float3 baseColor = make_float3(baseColor4.x, baseColor4.y, baseColor4.z);
  float baseBrightness = (baseColor.x + baseColor.y + baseColor.z) / 3.0f;
  
  std::cout << "Material Properties:" << std::endl;
  std::cout << "  Metallic Factor: " << metallicFactor << std::endl;
  std::cout << "  Roughness Factor: " << roughnessFactor << std::endl;
  std::cout << "  Base Color Brightness: " << baseBrightness << std::endl;
  
  bool isCompatible = true;
  recommendedBrightness = baseBrightness;
  
  // Note potential issues without making adjustments
  if (metallicFactor > 0.5f && roughnessFactor < 0.3f) {
    std::cout << "ℹ️  NOTE: High metallic + low roughness detected" << std::endl;
  }
  
  if (baseBrightness < 0.5f) {
    std::cout << "ℹ️  NOTE: Low base brightness detected" << std::endl;
  }
  
  if (baseBrightness > 2.0f) {
    std::cout << "ℹ️  NOTE: Very bright base color detected" << std::endl;
  }
  
  std::cout << "✓ Material analysis complete - using original properties" << std::endl;
  std::cout << "=== Compatibility Check Complete ===" << std::endl;
  
  return true; // Always compatible since we don't adjust
}


void Application::selectAppropriateGroundMaterial(MaterialData& groundMaterial) {
  std::cout << "\n=== Selecting Appropriate Ground Material ===" << std::endl;
  
  // Reset to default values
  groundMaterial.baseColorFactor = {0.8f, 0.7f, 0.6f, 1.0f}; // Brighter default color for better shadow visibility
  groundMaterial.metallicFactor = 0.0f;
  groundMaterial.roughnessFactor = 1.0f;
  
  // Use default material properties without adjustments
  std::cout << "Using default ground material properties" << std::endl;
  
  std::cout << "Selected Material Properties:" << std::endl;
  std::cout << "  Base Color: (" << groundMaterial.baseColorFactor.x << ", " 
            << groundMaterial.baseColorFactor.y << ", " << groundMaterial.baseColorFactor.z << ")" << std::endl;
  std::cout << "  Metallic Factor: " << groundMaterial.metallicFactor << std::endl;
  std::cout << "  Roughness Factor: " << groundMaterial.roughnessFactor << std::endl;
  std::cout << "=== Material Selection Complete ===" << std::endl;
}

void Application::saveRenderChannels(const std::string& baseFilename) {
  std::cout << "\n=== Saving Multi-Channel Render Output ===" << std::endl;
  
  // Copy device buffers to host
  const size_t numElements = size_t(m_resolution.x) * size_t(m_resolution.y);
  
  CUDA_CHECK(cudaMemcpy(m_bufferHost, m_launchParameters.bufferAccum, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferAlbedo, m_launchParameters.bufferAlbedo, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferDepth, m_launchParameters.bufferDepth, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferNormal, m_launchParameters.bufferNormal, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferRoughness, m_launchParameters.bufferRoughness, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferMetallic, m_launchParameters.bufferMetallic, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_bufferMask, m_launchParameters.bufferMask, 
                       numElements * sizeof(float4), cudaMemcpyDeviceToHost));
  
  
  // Save each channel (noisy versions)
  saveImageChannel(baseFilename + "_rgb_noisy.png", m_bufferHost, m_resolution.x, m_resolution.y, "RGB (Noisy)");
  saveImageChannel(baseFilename + "_albedo.png", m_bufferAlbedo, m_resolution.x, m_resolution.y, "Albedo");
  saveImageChannel(baseFilename + "_depth.png", m_bufferDepth, m_resolution.x, m_resolution.y, "Depth");
  saveImageChannel(baseFilename + "_normal.png", m_bufferNormal, m_resolution.x, m_resolution.y, "Normal");
  saveImageChannel(baseFilename + "_roughness.png", m_bufferRoughness, m_resolution.x, m_resolution.y, "Roughness");
  saveImageChannel(baseFilename + "_metallic.png", m_bufferMetallic, m_resolution.x, m_resolution.y, "Metallic");
  saveImageChannel(baseFilename + "_mask.png", m_bufferMask, m_resolution.x, m_resolution.y, "Mask");
  
  
  // Apply denoising to RGB and save denoised version
  if (m_enableDenoiser && m_denoiser && m_d_denoisedBuffer) {
    // Apply denoising
    invokeDenoiser();
    CUDA_CHECK(cudaStreamSynchronize(m_cudaStream)); // Wait for denoising to complete
    
    // Copy denoised buffer to host
    float4* denoisedHost = new float4[numElements];
    CUDA_CHECK(cudaMemcpy(denoisedHost, reinterpret_cast<void*>(m_d_denoisedBuffer), 
                         numElements * sizeof(float4), cudaMemcpyDeviceToHost));
    
    saveImageChannel(baseFilename + "_rgb_denoised.png", denoisedHost, m_resolution.x, m_resolution.y, "RGB (Denoised)");
    
    delete[] denoisedHost;
  }
  
  std::cout << "=== Multi-Channel Save Complete ===" << std::endl;
}

void Application::saveImageChannel(const std::string& filename, const float4* data, int width, int height, const std::string& channelName) {
  std::cout << "Saving " << channelName << " channel: " << filename << std::endl;
  
  ILuint imageId = 0;
  
  if (filename.substr(filename.length() - 4) == ".hdr") {
    // Save as HDR format
    imageId = ilTexImage(width, height, 1, 4, IL_RGBA, IL_FLOAT, (void*)data);
    if (imageId) {
      ilSave(IL_HDR, filename.c_str());
      ilDeleteImages(1, &imageId);
    }
  } else {
    // Save as PNG format (convert to LDR)
    std::vector<unsigned char> ldrData(width * height * 3);
    
    // Debug output removed - data is correct
    
    for (int i = 0; i < width * height; ++i) {
      float r = data[i].x;
      float g = data[i].y;
      float b = data[i].z;
      
      // Special processing for different channel types
      if (channelName == "Depth") {
        // Depth is camera-space depth: Dist = -t_hit * dot(V, cameraW)
        // Camera-space depth can be negative (behind camera), positive (in front)
        // Typical values for camera-space depth are smaller than world-space distance
        // For better visualization, use a smaller maxDepth for camera-space depth
        float maxDepth = 50.0f; // Reduced for camera-space depth (typically 1-30 for most scenes)
        
        // Camera-space depth: positive values are in front of camera
        // Normalize by taking absolute value and clamping
        float absDepth = fabsf(r);
        float normalizedDepth = fminf(absDepth / maxDepth, 1.0f);
        
        // Invert so closer objects are brighter, and handle negative depths
        // For camera-space depth, negative means behind camera, treat as far
        if (r < 0.0f) {
          normalizedDepth = 1.0f; // Behind camera = far = dark
        }
        r = g = b = 1.0f - normalizedDepth; // Invert so closer objects are brighter
      } else if (channelName == "Mask") {
        // Mask is single channel in x component, already in [0,1] range
        r = g = b = r; // Use mask value for all channels (grayscale)
      } else if (channelName == "Normal") {
        // Normal channel is already in [0,1] range (converted in raygen)
        // No additional conversion needed
        // Values are already normalized from [-1,1] to [0,1]
      } else if (channelName == "Roughness" || channelName == "Metallic") {
        // These are already in [0,1] range, just use the first component
        r = g = b = r;
      } else if (channelName == "RGB" || channelName == "RGB (Noisy)" || channelName == "RGB (Denoised)") {
        // Use the same tonemapping as the display to ensure consistency
        const float invGamma = 1.0f / m_gamma;
        const float3 colorBalance = make_float3(m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
        const float invWhitePoint = m_brightness / m_whitePoint;
        const float burnHighlights = m_burnHighlights;
        const float crushBlacks = m_crushBlacks + m_crushBlacks + 1.0f;
        const float saturation = m_saturation;
        
        float3 hdrColor = make_float3(r, g, b);
        
        // Apply the same tonemapping pipeline as display
        float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
        ldrColor *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);
        
        float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
        ldrColor = lerp(make_float3(luminance), ldrColor, saturation);
        ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
        
        luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
        if (luminance < 1.0f) {
          const float3 crushed = powf(ldrColor, crushBlacks);
          ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
          ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
        }
        ldrColor = clamp(powf(ldrColor, invGamma), 0.0f, 1.0f);
        
        r = ldrColor.x;
        g = ldrColor.y;
        b = ldrColor.z;
      } else if (channelName == "Albedo") {
        // Diffuse/Albedo channel - apply simple gamma correction for better visualization
        // This represents the diffuse lighting result (Tutorial 1 style)
        r = powf(fmaxf(r, 0.0f), 1.0f / 2.2f);
        g = powf(fmaxf(g, 0.0f), 1.0f / 2.2f);
        b = powf(fmaxf(b, 0.0f), 1.0f / 2.2f);
      }
      // Other channels use original values without tone mapping
      
      // Debug output removed - data processing is correct
      
      // Convert to 8-bit and clamp
      ldrData[i * 3 + 0] = (unsigned char)(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
      ldrData[i * 3 + 1] = (unsigned char)(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
      ldrData[i * 3 + 2] = (unsigned char)(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    }
    
    imageId = ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, ldrData.data());
    if (imageId) {
      ilSave(IL_PNG, filename.c_str());
      ilDeleteImages(1, &imageId);
    }
  }
  
  if (!imageId) {
    std::cerr << "Failed to save " << channelName << " channel: " << filename << std::endl;
  }
}

void Application::initTrackball() {
  // Initialize default camera
  m_camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
  m_camera.setUp(glm::vec3(0.0f, 1.0f, 0.0f));
  m_camera.setFovY(90.0f); // Even wider FOV for better asset visibility
  m_camera.setAspectRatio(1.0f);

  // Position camera to focus on objects (center area) rather than entire ground
  if (m_sceneExtent.isValid()) {
    glm::vec3 center = m_sceneExtent.getCenter();
    
    // Controlled randomization for camera position (matching reference images)
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Distance: balanced to see full ground with good object focus
    std::uniform_real_distribution<float> distanceDist(0.45f, 0.55f);
    float distanceMultiplier = distanceDist(gen);
    float focusDistance = distanceMultiplier * m_sceneExtent.getMaxDimension();
    
    // Azimuth: full 360° rotation for variety
    std::uniform_real_distribution<float> azimuthDist(0.0f, 2.0f * M_PI);
    float azimuth = azimuthDist(gen);
    
    // Elevation angle: 5-15 degrees from horizontal plane (very low angle for better asset visibility)
    // tan(θ) = elevationMultiplier / (1 - elevationMultiplier)
    // θ=5°: elevationMultiplier≈0.08, θ=15°: elevationMultiplier≈0.21
    std::uniform_real_distribution<float> elevationDist(0.08f, 0.21f);
    float elevationMultiplier = elevationDist(gen);
    
    // Calculate camera position using spherical coordinates
    float horizontalDistance = focusDistance * (1.0f - elevationMultiplier);
    float x = center.x + horizontalDistance * cos(azimuth);
    float z = center.z + horizontalDistance * sin(azimuth);
    float y = center.y + focusDistance * elevationMultiplier;
    
    // Calculate actual elevation angle for display
    float elevationAngle = atan(elevationMultiplier / (1.0f - elevationMultiplier)) * 180.0f / M_PI;
    
    glm::vec3 cameraPos(x, y, z);
    
    std::cout << "Camera setup: azimuth=" << (azimuth * 180.0f / M_PI) << "°, "
              << "distance=" << focusDistance << ", height=" << y 
              << ", elevation angle=" << elevationAngle << "° (from horizontal)" << std::endl;

    m_camera.setPosition(cameraPos);
    m_camera.setLookat(center);
  }

  // Setup trackball controller
  m_trackball.setCamera(&m_camera);
  m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
  m_trackball.setGimbalLock(m_isLockedGimbal);
}

void Application::initLaunchParameters() {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_launchParameters), sizeof(LaunchParameters)));

  m_launchParameters.handle = m_ias; // Root traversable handle of the scene.

  // Output buffer for the rendered image (HDR linear color).
  // This is initialized inside updateBuffers() depending on the m_interop state.
  m_launchParameters.bufferAccum = nullptr;
  
  // Multi-channel rendering buffers (initialized in updateBuffers())
  m_launchParameters.bufferAlbedo = nullptr;
  m_launchParameters.bufferDepth = nullptr;
  m_launchParameters.bufferNormal = nullptr;
  m_launchParameters.bufferRoughness = nullptr;
  m_launchParameters.bufferMetallic = nullptr;
  m_launchParameters.bufferMask = nullptr;
  m_launchParameters.targetAssetIndex = -1; // No target asset by default

  // Output buffer for the picked material index.
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferPicking), sizeof(int)));

  m_launchParameters.resolution = m_resolution;    // Independent of the window client area.
  m_launchParameters.picking = make_float2(-1.0f); // No picking ray.
  m_launchParameters.pathLengths = make_int2(2, 12);  // Maximum path length for high-quality lighting and reflections
  m_launchParameters.iteration =
    0u; // Sub-frame number for the progressive accumulation of results.
  m_launchParameters.sceneEpsilon = m_epsilonFactor * SCENE_EPSILON_SCALE;
  m_launchParameters.directLighting = (m_useDirectLighting) ? 1 : 0;
  m_launchParameters.ambientOcclusion = (m_useAmbientOcclusion) ? 1 : 0;
  m_launchParameters.showEnvironment = (m_showEnvironment) ? 1 : 0;
  m_launchParameters.forceUnlit = (m_forceUnlit) ? 1 : 0;
  
  m_launchParameters.textureSheenLUT =
    (m_texSheenLUT != nullptr) ? m_texSheenLUT->getTextureObject() : 0;

  // Initialize light definitions pointer (will be set properly after loadHDREnvironmentLight)
  m_launchParameters.lightDefinitions = nullptr; // Will be set in updateLaunchParameters()
  m_launchParameters.numLights = 0; // Will be set in updateLaunchParameters()
  
  // Initialize asset comparison parameters

  // All dirty flags are set here and the first render() call will take care to allocate and update
  // all necessary resources.
}

void Application::updateLaunchParameters() {
  // This is called after acceleration structures have been rebuilt.
  m_launchParameters.handle = m_ias; // Update the top-level IAS handle.

  // Update light definitions on device - CRITICAL for proper HDR switching
  if (!m_lightDefinitions.empty()) {
    // Free old device memory if it exists
    if (m_d_lightDefinitions != 0) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_lightDefinitions)));
    }
    
    // Allocate new device memory for current light definitions
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_lightDefinitions), 
                          sizeof(LightDefinition) * m_lightDefinitions.size()));
    
    // Copy current light definitions to device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_lightDefinitions), 
                          m_lightDefinitions.data(),
                          sizeof(LightDefinition) * m_lightDefinitions.size(), 
                          cudaMemcpyHostToDevice));
    
    // Update launch parameters with new device pointer
    m_launchParameters.lightDefinitions = reinterpret_cast<LightDefinition*>(m_d_lightDefinitions);
    m_launchParameters.numLights = static_cast<int>(m_lightDefinitions.size());
    
    std::cout << "Updated light definitions on device: " << m_launchParameters.numLights << " lights" << std::endl;
  } else {
    std::cerr << "WARNING: No light definitions available, using null pointer" << std::endl;
    m_launchParameters.lightDefinitions = nullptr;
    m_launchParameters.numLights = 0;
  }


  m_launchParameters.iteration = 0u; // Restart accumulation.
  
  // Update environment display setting
  m_launchParameters.showEnvironment = (m_showEnvironment) ? 1 : 0;
  
  // Update target asset index for mask generation
  m_launchParameters.targetAssetIndex = m_targetAssetIndex;
}


void Application::updateCamera() {
  // Sync trackball controller with camera
  if (m_trackball.getCamera() != &m_camera) {
    m_trackball.setCamera(&m_camera);

    // Initialize trackball reference frame
    m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                                  glm::vec3(0.0f, 0.0f, 1.0f));

    // Keep models upright when orbiting
    m_trackball.setGimbalLock(m_isLockedGimbal);
  }

  // Update camera aspect ratio based on resolution
  m_camera.setAspectRatio(static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y));

  // Set camera type: 0 == orthographic, 1 == perspective
  m_launchParameters.cameraType = (0.0f < m_camera.getFovY()) ? 1 : 0;

  // Update camera position
  glm::vec3 P = m_camera.getPosition();
  m_launchParameters.cameraP = make_float3(P.x, P.y, P.z);

  // Get camera orientation vectors
  glm::vec3 U, V, W;
  m_camera.getUVW(U, V, W);

  // Convert to CUDA float3 vector types
  m_launchParameters.cameraU = make_float3(U.x, U.y, U.z);
  m_launchParameters.cameraV = make_float3(V.x, V.y, V.z);
  m_launchParameters.cameraW = make_float3(W.x, W.y, W.z);

  m_launchParameters.iteration = 0; // Restart accumulation
  m_camera.setIsDirty(false);
}

void Application::update() {
  MY_ASSERT(!m_isDirtyResolution && m_hdrTexture != 0);

  switch (m_interop) {
  case INTEROP_OFF:
    // Copy the GPU local render buffer into host and update the HDR texture image from there.
    MY_ASSERT(m_bufferHost != nullptr);
    
    CUDA_CHECK(cudaMemcpy((void*)m_bufferHost, m_launchParameters.bufferAccum,
                          m_resolution.x * m_resolution.y * sizeof(float4),
                          cudaMemcpyDeviceToHost));
    
    // Copy the host buffer to the OpenGL texture image (slowest path, most portable).
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y,
                    GL_RGBA, GL_FLOAT, m_bufferHost); // RGBA32F
    break;

  case INTEROP_PBO:
    // The image was rendered into the linear PBO buffer directly. Just upload to the block-linear
    // texture.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)m_resolution.x, (GLsizei)m_resolution.y,
                    GL_RGBA, GL_FLOAT,
                    (GLvoid*)0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    break;

  case INTEROP_TEX: {
    CUarray dstArray = nullptr;

    // Map the Texture object directly and copy the output buffer.
    CU_CHECK(cuGraphicsMapResources(1, &m_cudaGraphicsResource,
                                    m_cudaStream)); // This is an implicit cuSynchronizeStream().
    CU_CHECK(cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0,
                                                 0)); // arrayIndex = 0, mipLevel = 0

    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    params.srcDevice = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferAccum);
    params.srcPitch = m_resolution.x * sizeof(float4); // RGBA32F
    params.srcHeight = m_resolution.y;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray = dstArray;
    params.WidthInBytes = m_resolution.x * sizeof(float4);
    params.Height = m_resolution.y;
    params.Depth = 1;

    CU_CHECK(cuMemcpy3D(&params)); // Copy from linear to array layout.

    CU_CHECK(cuGraphicsUnmapResources(1, &m_cudaGraphicsResource,
                                      m_cudaStream)); // This is an implicit cuSynchronizeStream().
  } break;

  case INTEROP_IMG:
    // Nothing to do. Renders into the m_hdrTexture surface object directly.
    break;
  }
}

void Application::updateBufferHost() {
  update();

  // After the update() call, the m_hdrTexture contains the linear HDR image.
  // When interop is off, the m_bufferHost already contains the current linear HDR image data as
  // well.
  if (m_interop != INTEROP_OFF) {
    // Read the m_hdrTexture image into the m_bufferHost.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (GLvoid*)m_bufferHost);
  }
}

bool Application::screenshot(const bool tonemap) {
  updateBufferHost(); // Make sure m_bufferHost contains the linear HDR image data.

  ILboolean hasImage = false;

  std::ostringstream path;

  path << "img_gltf_" << utils::getDateTime();

  unsigned int imageID;

  ilGenImages(1, (ILuint*)&imageID);

  ilBindImage(imageID);
  ilActiveImage(0);
  ilActiveFace(0);

  ilDisable(IL_ORIGIN_SET);

  if (tonemap) {
    path << ".png"; // Store a tonemapped RGB8 *.png image

    if (ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 3, IL_RGB,
                   IL_UNSIGNED_BYTE, nullptr)) {
      uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

      const float invGamma = 1.0f / m_gamma;
      const float3 colorBalance = make_float3(m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      const float invWhitePoint = m_brightness / m_whitePoint;
      const float burnHighlights = m_burnHighlights;
      const float crushBlacks = m_crushBlacks + m_crushBlacks + 1.0f;
      const float saturation = m_saturation;

      for (int y = 0; y < m_launchParameters.resolution.y; ++y) {
        for (int x = 0; x < m_launchParameters.resolution.x; ++x) {
          const int idx = m_launchParameters.resolution.x * y + x;

          // Tonemapper. // PERF Add a native CUDA kernel doing this.
          float3 hdrColor = make_float3(m_bufferHost[idx]);

          float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
          ldrColor *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);

          float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          ldrColor = lerp(make_float3(luminance), ldrColor,
                          saturation); // This can generate negative values for saturation > 1.0f!
          ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.

          luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          if (luminance < 1.0f) {
            const float3 crushed = powf(ldrColor, crushBlacks);
            ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
            ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
          }
          ldrColor =
            clamp(powf(ldrColor, invGamma), 0.0f, 1.0f); // Saturate, clamp to range [0.0f, 1.0f].

          dst[idx] =
            make_uchar3((unsigned char)(ldrColor.x * 255.0f), (unsigned char)(ldrColor.y * 255.0f),
                        (unsigned char)(ldrColor.z * 255.0f));
        }
      }
      hasImage = true;
    }
  } else {
    path << ".hdr"; // Store the float4 linear output buffer as *.hdr image.

    hasImage = ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 4,
                          IL_RGBA, IL_FLOAT, (void*)m_bufferHost);
  }

  if (hasImage) {
    ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite

    std::string filename = path.str();
    utils::convertPath(filename);

    if (ilSaveImage((const ILstring)filename.c_str())) {
      ilDeleteImages(1, &imageID);

      std::cout << filename
                << '\n'; // Print out filename to indicate that a screenshot has been taken.
      return true;
    }
  }

  // There was an error when reaching this code.
  ILenum error = ilGetError(); // DEBUG
  std::cerr << "ERROR: screenshot() failed with IL error " << error << '\n';

  while (ilGetError() != IL_NO_ERROR) // Clean up errors.
  {
  }

  // Free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);

  return false;
}

// Init (if needed) and scale the font depending on screen DPI.
void Application::updateFonts() {
  // Create or update the font
  ImGuiIO& io = ImGui::GetIO();

  // Only calculate font scale if we have a window context
  float fontScale = 1.0f; // Default scale for headless mode
  if (m_window != nullptr) {
    fontScale = utils::getFontScale();
  }
  
  if (fontScale == m_fontScale && m_font != nullptr) {
    return;
  }

  // change of DPI detected or no font yet
  m_fontScale = fontScale;
  io.FontGlobalScale = m_fontScale;
  io.FontAllowUserScaling = true; // enable scaling with ctrl + wheel.
  std::cerr << "FontGlobalScale " << io.FontGlobalScale << std::endl;

#if defined(_WIN32)
  static const char* fontName{"C:/Windows/Fonts/arialbd.ttf"};
#else
  // works on Ubuntu Linux
  static const char* fontName{"/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"};
#endif

  // create and/or scale the font (only if we have a window/OpenGL context)
  if (m_window != nullptr) {
    if (m_font == nullptr) {
      // load the font and create the texture
      io.Fonts->AddFontDefault();
      std::cout << "Loading font " << fontName << std::endl;
      m_font = io.Fonts->AddFontFromFileTTF(fontName, 13.0f);
      glCreateTextures(GL_TEXTURE_2D, 1, &m_fontTexture);
    }

    if (m_font != nullptr) {
      // update the texture with scaled font data
      unsigned char* pixels = nullptr;
      int texW, texH;
      io.Fonts->GetTexDataAsRGBA32(&pixels, &texW, &texH);

      // DONT glBindTextures(0, 1, &m_fontTexture); or device update will fail
      glTextureStorage2D(m_fontTexture, 1, GL_RGBA8, texW, texH);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTextureSubImage2D(m_fontTexture, 0, 0, 0, texW, texH, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

#pragma warning(push)
#pragma warning(disable : 4312)
      io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(m_fontTexture));
#pragma warning(pop)
    } else {
      std::cerr << "ERROR can't load font " << fontName << std::endl;
    }
  }
}

unsigned int Application::getBuildFlags() const {
  unsigned int flags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  flags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  return flags;
}

void Application::cameraTranslate(float dx, float dy, float dz) {
  MY_ASSERT(m_sceneExtent.isValid());
  const float scale = m_sceneExtent.getDiameter() * CameraSpeedFraction;
  MY_ASSERT(0.0f != scale);

  const glm::vec3 transl{scale * dx, scale * dy, scale * dz};

  // Move camera and lookat point together
  const auto fwd = m_camera.getDirection();
  const auto right = m_camera.getRight();
  const auto up = m_camera.getUp();
  const auto delta = (fwd * transl.z) + (right * transl.x) + (up * transl.y);

  auto pos = m_camera.getPosition();
  auto lookat = m_camera.getLookat();
  pos += delta;
  lookat += delta;

  m_camera.setPosition(pos);
  m_camera.setLookat(lookat);
}

// Asset comparison rendering functionality
void Application::renderAssetComparison() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Starting Asset Comparison Rendering" << std::endl;
  std::cout << "========================================" << std::endl;
  
  // Step 1: Load HDRI environments - Use HDR1 from command line and fixed HDR2
  std::vector<std::string> hdriPaths;
  
  // HDR1: Get from command line arguments
  std::string hdr1Path = m_options.getHdr1Path();
  std::cout << "HDR1: Using HDR from command line: " << hdr1Path << std::endl;
  hdriPaths.push_back(hdr1Path);
  
  // HDR2: Get from command line arguments
  std::string hdr2Path = m_options.getHdr2Path();
  std::cout << "HDR2: Using HDR from command line: " << hdr2Path << std::endl;
  hdriPaths.push_back(hdr2Path);
  
  // Step 2: Select target asset
  selectTargetAsset();
  if (m_targetAssetIndex == -1) {
    std::cout << "Error: No target asset selected. Aborting comparison." << std::endl;
    return;
  }
  
  // Step 3: Render with target asset in both HDRI environments
  std::cout << "\n=== Rendering with Target Asset ===" << std::endl;
  for (size_t i = 0; i < hdriPaths.size(); ++i) {
    std::cout << "Loading HDRI " << (i + 1) << ": " << hdriPaths[i] << std::endl;
    
    // Load the HDRI environment
    if (m_picEnv != nullptr) {
      delete m_picEnv;
    }
    m_picEnv = new Picture();
    
    if (!m_picEnv->load(hdriPaths[i], IMAGE_FLAG_2D)) {
      std::cout << "Failed to load HDRI " << (i + 1) << ", skipping..." << std::endl;
      continue;
    }
    
    // Create texture from HDRI
    if (m_texEnv != nullptr) {
      delete m_texEnv;
    }
    m_texEnv = new Texture(m_allocator);
    if (!m_texEnv->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) {
      std::cout << "Failed to create texture from HDRI " << (i + 1) << ", skipping..." << std::endl;
      continue;
    }
    
    // Update environment light using the manually loaded texture
    LightDefinition light = createSphericalEnvironmentLightFromExistingTexture();
    m_lightDefinitions.clear();
    m_lightDefinitions.push_back(light);
    
    // Update launch parameters to ensure consistent lighting
    updateLaunchParameters();
    
    // Render with target asset visible
    showTargetAsset();
    buildInstanceAccel(true); // Rebuild acceleration structure
    
    // Render multiple samples for better quality
    for (int sample = 0; sample < m_comparisonSamples; ++sample) {
      render();
    }
    
    // Save images with target asset
    std::string baseFilename = "hdri" + std::to_string(i + 1) + "_with_asset";
    renderWithCurrentEnvironment(baseFilename, hdriPaths[i]);
  }
  
  // Step 4: Hide target asset and render again in both HDRI environments
  std::cout << "\n=== Rendering without Target Asset ===" << std::endl;
  hideTargetAsset();
  buildInstanceAccel(true); // Rebuild acceleration structure
  
  for (size_t i = 0; i < hdriPaths.size(); ++i) {
    std::cout << "Loading HDRI " << (i + 1) << " again: " << hdriPaths[i] << std::endl;
    
    // Target asset is already hidden, no need to call hideTargetAsset() again
    
    // Load the HDRI environment again
    if (m_picEnv != nullptr) {
      delete m_picEnv;
    }
    m_picEnv = new Picture();
    
    if (!m_picEnv->load(hdriPaths[i], IMAGE_FLAG_2D)) {
      std::cout << "Failed to load HDRI " << (i + 1) << ", skipping..." << std::endl;
      continue;
    }
    
    // Create texture from HDRI
    if (m_texEnv != nullptr) {
      delete m_texEnv;
    }
    m_texEnv = new Texture(m_allocator);
    if (!m_texEnv->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) {
      std::cout << "Failed to create texture from HDRI " << (i + 1) << ", skipping..." << std::endl;
      continue;
    }
    
    // Update environment light using existing texture
    LightDefinition light = createSphericalEnvironmentLightFromExistingTexture();
    m_lightDefinitions.clear();
    m_lightDefinitions.push_back(light);
    
    // Update launch parameters to ensure consistent lighting
    updateLaunchParameters();
    
    
    // Render multiple samples for better quality
    for (int sample = 0; sample < m_comparisonSamples; ++sample) {
      render();
    }
    
    // Save images without target asset
    std::string baseFilename = "hdri" + std::to_string(i + 1) + "_without_asset";
    renderWithCurrentEnvironment(baseFilename, hdriPaths[i]);
  }
  
  // Step 5: Restore target asset visibility
  std::cout << "Step 5: Restoring target asset visibility..." << std::endl;
  showTargetAsset();
  std::cout << "Target asset restored, rebuilding acceleration structure..." << std::endl;
  buildInstanceAccel(true);
  std::cout << "Acceleration structure rebuilt successfully" << std::endl;
  
  
  std::cout << "\n========================================" << std::endl;
  std::cout << "Asset Comparison Rendering Complete" << std::endl;
  std::cout << "Generated 4 sets of images in separate folders:" << std::endl;
  std::cout << "- comparison_output/hdri1_with_asset/" << std::endl;
  std::cout << "- comparison_output/hdri1_without_asset/" << std::endl;
  std::cout << "- comparison_output/hdri2_with_asset/" << std::endl;
  std::cout << "- comparison_output/hdri2_without_asset/" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "Each folder contains:" << std::endl;
  std::cout << "- RGB, Albedo, Depth, Normal, Roughness, Metallic images" << std::endl;
  std::cout << "- Corresponding HDRI environment file" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "========================================\n" << std::endl;
}

bool Application::validateHDRIFile(const std::string& hdrPath) {
  // Create a temporary Picture object to test the HDRI file
  Picture* testPic = new Picture();
  bool isValid = false;
  
  try {
    if (testPic->load(hdrPath, IMAGE_FLAG_2D)) {
      // Check if the loaded image has valid data
      if (testPic->getNumberOfImages() > 0) {
        const Image* image = testPic->getImageLevel(0, 0); // Get first image, first level
        if (image != nullptr && image->m_pixels != nullptr) {
          // Basic safety checks
          int width = image->m_width;
          int height = image->m_height;
          
          if (width > 0 && height > 0 && width < 10000 && height < 10000) {
            // Just check if we can access the data without crashing
            // Don't do detailed pixel validation to avoid being too strict
            isValid = true;
          } else {
            std::cout << "HDRI file " << hdrPath << " has invalid dimensions, skipping..." << std::endl;
          }
        } else {
          std::cout << "HDRI file " << hdrPath << " has no valid image data, skipping..." << std::endl;
        }
      } else {
        std::cout << "HDRI file " << hdrPath << " has no images, skipping..." << std::endl;
      }
    } else {
      std::cout << "Failed to load HDRI file " << hdrPath << ", skipping..." << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "Exception while validating HDRI file " << hdrPath << ": " << e.what() << ", skipping..." << std::endl;
  } catch (...) {
    std::cout << "Unknown exception while validating HDRI file " << hdrPath << ", skipping..." << std::endl;
  }
  
  delete testPic;
  return isValid;
}

// Analyze HDR directional strength for better shadow generation
float Application::analyzeHDRDirectionality(const std::string& hdrPath) {
  try {
    Picture tempPic;
    if (!tempPic.load(hdrPath, IMAGE_FLAG_2D)) {
      return 0.0f; // Invalid file
    }

    const Image* image = tempPic.getImageLevel(0, 0);
    if (!image || !image->m_pixels) {
      return 0.0f; // No valid image data
    }

    int width = image->m_width;
    int height = image->m_height;
    int pixelCount = width * height;

    // Calculate directional strength by analyzing brightness distribution
    float totalBrightness = 0.0f;
    float maxBrightness = 0.0f;
    float brightPixelCount = 0.0f;
    float directionalScore = 0.0f;

    // For HDR files, pixels are stored as RGBE format (4 bytes per pixel)
    // or as float RGB (12 bytes per pixel). We need to handle both cases.
    if (image->m_bpp == 4) {
      // RGBE format: 4 bytes per pixel (R, G, B, E)
      for (int i = 0; i < pixelCount; ++i) {
        unsigned char r = image->m_pixels[i * 4 + 0];
        unsigned char g = image->m_pixels[i * 4 + 1];
        unsigned char b = image->m_pixels[i * 4 + 2];
        unsigned char e = image->m_pixels[i * 4 + 3];

        // Convert RGBE to float RGB
        if (e == 0) {
          // Special case: all zeros
          continue;
        }

        float scale = powf(2.0f, (float)e - 128.0f);
        float fr = (r + 0.5f) * scale / 256.0f;
        float fg = (g + 0.5f) * scale / 256.0f;
        float fb = (b + 0.5f) * scale / 256.0f;

        // Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        float luminance = 0.299f * fr + 0.587f * fg + 0.114f * fb;
        totalBrightness += luminance;
        
        if (luminance > maxBrightness) {
          maxBrightness = luminance;
        }
        
        // Count pixels above threshold (indicating strong light sources)
        if (luminance > 0.5f) {
          brightPixelCount += 1.0f;
        }
      }
    } else if (image->m_bpp == 12) {
      // Float RGB format: 12 bytes per pixel (3 floats)
      float* floatPixels = reinterpret_cast<float*>(image->m_pixels);
      for (int i = 0; i < pixelCount; ++i) {
        float r = floatPixels[i * 3 + 0];
        float g = floatPixels[i * 3 + 1];
        float b = floatPixels[i * 3 + 2];

        // Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
        totalBrightness += luminance;
        
        if (luminance > maxBrightness) {
          maxBrightness = luminance;
        }
        
        // Count pixels above threshold (indicating strong light sources)
        if (luminance > 0.5f) {
          brightPixelCount += 1.0f;
        }
      }
    } else {
      // Unknown format, return 0
      return 0.0f;
    }

    float avgBrightness = totalBrightness / pixelCount;
    
        // Calculate directional score based on:
        // 1. High maximum brightness (strong light sources)
        // 2. Low bright pixel count ratio (concentrated light sources)
        // 3. High average brightness (overall bright environment)
        // 4. Brightness variance (more variance = more directional)
        
        float brightPixelRatio = brightPixelCount / pixelCount;
        
        // Calculate brightness variance for better directionality detection
        float brightnessVariance = 0.0f;
        if (image->m_bpp == 4) {
          // RGBE format
          for (int i = 0; i < pixelCount; ++i) {
            unsigned char r = image->m_pixels[i * 4 + 0];
            unsigned char g = image->m_pixels[i * 4 + 1];
            unsigned char b = image->m_pixels[i * 4 + 2];
            unsigned char e = image->m_pixels[i * 4 + 3];
            
            if (e == 0) continue;
            
            float scale = powf(2.0f, (float)e - 128.0f);
            float fr = (r + 0.5f) * scale / 256.0f;
            float fg = (g + 0.5f) * scale / 256.0f;
            float fb = (b + 0.5f) * scale / 256.0f;
            float luminance = 0.299f * fr + 0.587f * fg + 0.114f * fb;
            
            float diff = luminance - avgBrightness;
            brightnessVariance += diff * diff;
          }
        } else if (image->m_bpp == 12) {
          // Float RGB format
          float* floatPixels = reinterpret_cast<float*>(image->m_pixels);
          for (int i = 0; i < pixelCount; ++i) {
            float r = floatPixels[i * 3 + 0];
            float g = floatPixels[i * 3 + 1];
            float b = floatPixels[i * 3 + 2];
            float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
            
            float diff = luminance - avgBrightness;
            brightnessVariance += diff * diff;
          }
        }
        brightnessVariance /= pixelCount;
        
        // Improved directional score with variance component
        // Higher variance indicates more concentrated lighting
        directionalScore = (maxBrightness * 0.3f) + 
                          ((1.0f - brightPixelRatio) * 0.25f) + 
                          (avgBrightness * 0.2f) +
                          (sqrtf(brightnessVariance) * 0.25f);
    
    return directionalScore;
  } catch (const std::exception& e) {
    std::cout << "Error analyzing HDR directionality: " << e.what() << std::endl;
    return 0.0f;
  }
}

// Analyze HDR brightness for debugging purposes
float Application::analyzeHDRBrightness(const std::string& hdrPath) {
  try {
    Picture tempPic;
    if (!tempPic.load(hdrPath, IMAGE_FLAG_2D)) {
      return 0.0f; // Invalid file
    }
    
    const Image* image = tempPic.getImageLevel(0, 0);
    if (!image || !image->m_pixels) {
      return 0.0f; // No valid image data
    }
    
    float totalBrightness = 0.0f;
    int pixelCount = image->m_width * image->m_height;
    
    // For HDR files, pixels are stored as RGBE format (4 bytes per pixel)
    // or as float RGB (12 bytes per pixel). We need to handle both cases.
    if (image->m_bpp == 4) {
      // RGBE format: 4 bytes per pixel (R, G, B, E)
      for (int i = 0; i < pixelCount; ++i) {
        unsigned char r = image->m_pixels[i * 4 + 0];
        unsigned char g = image->m_pixels[i * 4 + 1];
        unsigned char b = image->m_pixels[i * 4 + 2];
        unsigned char e = image->m_pixels[i * 4 + 3];
        
        // Convert RGBE to float RGB
        if (e == 0) {
          // Special case: all zeros
          continue;
        }
        
        float scale = powf(2.0f, (float)e - 128.0f);
        float fr = (r + 0.5f) * scale / 256.0f;
        float fg = (g + 0.5f) * scale / 256.0f;
        float fb = (b + 0.5f) * scale / 256.0f;
        
        // Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        float luminance = 0.299f * fr + 0.587f * fg + 0.114f * fb;
        totalBrightness += luminance;
      }
    } else if (image->m_bpp == 12) {
      // Float RGB format: 12 bytes per pixel (3 floats)
      float* floatPixels = reinterpret_cast<float*>(image->m_pixels);
      for (int i = 0; i < pixelCount; ++i) {
        float r = floatPixels[i * 3 + 0];
        float g = floatPixels[i * 3 + 1];
        float b = floatPixels[i * 3 + 2];
        
        // Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
        totalBrightness += luminance;
      }
    } else {
      // Unknown format, return 0
      return 0.0f;
    }
    
    return totalBrightness / pixelCount;
  } catch (const std::exception& e) {
    std::cout << "Error analyzing HDR brightness: " << e.what() << std::endl;
    return 0.0f;
  }
}

std::vector<std::string> Application::loadStrongerDirectionalHDR(int count) {
  std::vector<std::string> selectedHdrPaths;
  
  std::cout << "\n=== Selecting Stronger Directional HDR for HDR2 ===" << std::endl;
  
  // Scan HDRI directory for .hdr files
  std::vector<std::string> availableHdrFiles;
  try {
    std::filesystem::path hdriDir(m_path_hdri);
    if (std::filesystem::exists(hdriDir) && std::filesystem::is_directory(hdriDir)) {
      for (const auto& entry : std::filesystem::directory_iterator(hdriDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".hdr") {
          availableHdrFiles.push_back(entry.path().string());
        }
      }
    }
  } catch (const std::exception& e) {
    std::cout << "Error scanning HDRI directory: " << e.what() << std::endl;
  }
  
  if (availableHdrFiles.empty()) {
    std::cout << "No HDR files found in directory: " << m_path_hdri << std::endl;
    return selectedHdrPaths;
  }
  
  std::cout << "Found " << availableHdrFiles.size() << " HDR files in directory: " << m_path_hdri << std::endl;
  
  // Use stricter thresholds for stronger directional lighting
  const float directionalityThreshold = 0.4f; // Higher threshold for stronger directionality
  const float brightnessThreshold = 0.6f; // Higher threshold for brighter lighting
  
  std::cout << "Analyzing HDR directionality and brightness for stronger lighting..." << std::endl;
  std::cout << "Stricter thresholds - Directionality: " << directionalityThreshold << ", Brightness: " << brightnessThreshold << std::endl;
  
  // Analyze directionality and brightness
  std::vector<std::pair<std::string, std::pair<float, float>>> hdrScores; // {path, {directionality, brightness}}
  
  int analyzedCount = 0;
  int aboveThresholdCount = 0;
  
  for (const auto& hdrPath : availableHdrFiles) {
    float directionality = analyzeHDRDirectionality(hdrPath);
    float brightness = analyzeHDRBrightness(hdrPath);
    analyzedCount++;
    
    std::cout << "Testing stronger HDR " << analyzedCount << ": " << std::filesystem::path(hdrPath).filename().string() 
              << " (directionality: " << std::fixed << std::setprecision(3) << directionality 
              << ", brightness: " << std::fixed << std::setprecision(3) << brightness << ")";
    
    if (directionality >= directionalityThreshold && brightness >= brightnessThreshold) {
      hdrScores.push_back({hdrPath, {directionality, brightness}});
      aboveThresholdCount++;
      std::cout << " ✓ STRONG" << std::endl;
    } else {
      std::cout << " (below stricter thresholds)" << std::endl;
    }
    
    // Stop analyzing after checking reasonable number
    if (analyzedCount >= 50) {
      std::cout << "Stopping analysis after " << analyzedCount << " files" << std::endl;
      break;
    }
  }
  
  std::cout << "Stronger HDR analysis complete: " << aboveThresholdCount << " HDR files above stricter thresholds" << std::endl;
  
  if (hdrScores.empty()) {
    std::cout << "⚠ No HDR files meet stricter criteria, falling back to regular selection..." << std::endl;
    return loadRandomHDRIEnvironments(count);
  }
  
  // Sort by directionality (highest first) for strongest directional lighting
  std::sort(hdrScores.begin(), hdrScores.end(), 
            [](const std::pair<std::string, std::pair<float, float>>& a, 
               const std::pair<std::string, std::pair<float, float>>& b) {
              return a.second.first > b.second.first; // Sort by directionality descending
            });
  
  // Print top candidates
  std::cout << "\nTop stronger directional HDR candidates:" << std::endl;
  int printCount = std::min(5, (int)hdrScores.size());
  for (int i = 0; i < printCount; ++i) {
    std::cout << "  " << (i + 1) << ". " << std::filesystem::path(hdrScores[i].first).filename().string() 
              << " (directionality: " << std::fixed << std::setprecision(3) << hdrScores[i].second.first 
              << ", brightness: " << std::fixed << std::setprecision(3) << hdrScores[i].second.second << ")" << std::endl;
  }
  
  // Select the strongest directional HDR files
  int filesToSelect = std::min(count, (int)hdrScores.size());
  selectedHdrPaths.reserve(filesToSelect);
  
  std::cout << "\nSelected stronger directional HDR files:" << std::endl;
  for (int i = 0; i < filesToSelect; ++i) {
    selectedHdrPaths.push_back(hdrScores[i].first);
    std::cout << "  " << (i + 1) << ". " << std::filesystem::path(hdrScores[i].first).filename().string() 
              << " (directionality: " << std::fixed << std::setprecision(3) << hdrScores[i].second.first 
              << ", brightness: " << std::fixed << std::setprecision(3) << hdrScores[i].second.second << ")" << std::endl;
  }
  
  return selectedHdrPaths;
}

std::vector<std::string> Application::loadRandomHDRIEnvironments(int count) {
  std::cout << "\n=== Loading " << count << " Bright HDRI Environments (Debug Mode) ===" << std::endl;
  
  std::vector<std::string> selectedHdrPaths;
  
  // Scan HDRI directory for .hdr files
  std::vector<std::string> availableHdrFiles;
  try {
    std::filesystem::path hdriDir(m_path_hdri);
    if (std::filesystem::exists(hdriDir) && std::filesystem::is_directory(hdriDir)) {
      for (const auto& entry : std::filesystem::directory_iterator(hdriDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".hdr") {
          availableHdrFiles.push_back(entry.path().string());
        }
      }
    }
  } catch (const std::exception& e) {
    std::cout << "Error scanning HDRI directory: " << e.what() << std::endl;
    return selectedHdrPaths;
  }
  
  if (availableHdrFiles.empty()) {
    std::cout << "No HDR files found in directory: " << m_path_hdri << std::endl;
    return selectedHdrPaths;
  }
  
  std::cout << "Found " << availableHdrFiles.size() << " HDR files in directory" << std::endl;
  
  // Analyze brightness and select HDR files above threshold (good for shadow generation)
  std::vector<std::pair<std::string, float>> hdrBrightness;
  const float brightnessThreshold = 0.5f; // Lowered threshold for good shadow generation (was 0.8f)
  std::cout << "Analyzing HDR brightness for debug mode (threshold: " << brightnessThreshold << ")..." << std::endl;
  
  int analyzedCount = 0;
  int aboveThresholdCount = 0;
  
  for (const auto& hdrPath : availableHdrFiles) {
    float brightness = analyzeHDRBrightness(hdrPath);
    analyzedCount++;
    
    if (brightness >= brightnessThreshold) {
      hdrBrightness.push_back({hdrPath, brightness});
      aboveThresholdCount++;
      std::cout << "  " << std::filesystem::path(hdrPath).filename().string() 
                << " - brightness: " << std::fixed << std::setprecision(3) << brightness << " ✓" << std::endl;
    } else {
      std::cout << "  " << std::filesystem::path(hdrPath).filename().string() 
                << " - brightness: " << std::fixed << std::setprecision(3) << brightness << " (below threshold)" << std::endl;
    }
    
    // Stop analyzing after finding enough bright HDRs for the requested count or after checking reasonable number
    // For compare mode, we only need 2 HDRs, so stop early when we have enough
    int requiredCount = std::max(count, 2); // At least 2 for compare mode
    if (aboveThresholdCount >= requiredCount || analyzedCount >= 100) {
      std::cout << "Stopping analysis after " << analyzedCount << " files (found " << aboveThresholdCount << " above threshold, need " << requiredCount << ")" << std::endl;
      break;
    }
  }
  
  std::cout << "Analysis complete: " << aboveThresholdCount << " HDR files above threshold " << brightnessThreshold << std::endl;
  
  // Sort by brightness (highest first)
  std::sort(hdrBrightness.begin(), hdrBrightness.end(), 
            [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
              return a.second > b.second; // Sort by brightness descending
            });
  
  // Select the brightest HDR files
  int filesToSelect = std::min(count, (int)hdrBrightness.size());
  selectedHdrPaths.reserve(filesToSelect);
  
  std::cout << "\nSelected brightest HDR files for debug rendering:" << std::endl;
  for (int i = 0; i < filesToSelect; ++i) {
    selectedHdrPaths.push_back(hdrBrightness[i].first);
    std::cout << "  HDRI " << (i + 1) << ": " << std::filesystem::path(hdrBrightness[i].first).filename().string()
              << " (brightness: " << std::fixed << std::setprecision(3) << hdrBrightness[i].second << ")" << std::endl;
  }
  
  return selectedHdrPaths;
}

void Application::selectTargetAsset() {
  std::cout << "\n=== Selecting Target Asset ===" << std::endl;
  
  if (m_assets.empty()) {
    std::cout << "No assets available for selection" << std::endl;
    m_targetAssetIndex = -1;
    return;
  }
  
  // Smart selection: prefer assets that are likely to be visible in camera view
  std::random_device rd;
  std::mt19937 gen(rd());
  
  // Get camera position and direction for visibility check
  glm::vec3 cameraPos = m_camera.getPosition();
  glm::vec3 cameraLookat = m_camera.getLookat();
  glm::vec3 cameraDir = glm::normalize(cameraLookat - cameraPos);
  
  // Get camera UVW vectors for view frustum calculation
  glm::vec3 cameraU, cameraV, cameraW;
  m_camera.getUVW(cameraU, cameraV, cameraW);
  
  // Calculate camera FOV in radians
  float fovY = m_camera.getFovY() * M_PI / 180.0f; // Convert to radians
  float fovX = 2.0f * atan(tan(fovY * 0.5f) * m_camera.getAspectRatio()); // Calculate horizontal FOV
  
  std::cout << "Camera position: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
  std::cout << "Camera lookat: (" << cameraLookat.x << ", " << cameraLookat.y << ", " << cameraLookat.z << ")" << std::endl;
  std::cout << "Camera FOV: " << fovY * 180.0f / M_PI << "° (vertical), " << fovX * 180.0f / M_PI << "° (horizontal)" << std::endl;
  
  // Score assets based on visibility likelihood
  std::vector<std::pair<int, float>> assetScores; // {assetIndex, visibilityScore}
  
  for (size_t i = 0; i < m_assets.size(); ++i) {
    float visibilityScore = 0.0f;
    
    // Get asset position from its circle
    if (i < m_randomCircles.size()) {
      glm::vec3 assetPos = m_randomCircles[i].center;
      assetPos.y = 0.0f; // Ground level
      
      // Calculate distance from camera
      float distance = glm::length(assetPos - cameraPos);
      
      // Calculate angle from camera direction
      glm::vec3 toAsset = glm::normalize(assetPos - cameraPos);
      float dotProduct = glm::dot(cameraDir, toAsset);
      float angleFromCenter = acosf(std::max(-1.0f, std::min(1.0f, dotProduct)));
      
      // Check if asset is within camera view frustum
      bool inViewFrustum = true;
      
      // Project asset position to camera space
      glm::vec3 toAssetWorld = assetPos - cameraPos;
      
      // Calculate horizontal angle (around camera's up vector)
      glm::vec3 horizontalDir = glm::normalize(glm::vec3(toAssetWorld.x, 0.0f, toAssetWorld.z));
      glm::vec3 cameraHorizontalDir = glm::normalize(glm::vec3(cameraDir.x, 0.0f, cameraDir.z));
      float horizontalDot = glm::dot(horizontalDir, cameraHorizontalDir);
      float horizontalAngle = acosf(std::max(-1.0f, std::min(1.0f, horizontalDot)));
      
      // Calculate vertical angle (relative to camera's horizontal plane)
      float verticalAngle = atan2(toAssetWorld.y, glm::length(glm::vec3(toAssetWorld.x, 0.0f, toAssetWorld.z)));
      
      // Check if angles are within FOV bounds (use larger tolerance for better asset selection)
      if (horizontalAngle > fovX * 0.6f) { // Increased from 0.5f to 0.6f
        inViewFrustum = false;
      }
      if (std::abs(verticalAngle) > fovY * 0.6f) { // Increased from 0.5f to 0.6f
        inViewFrustum = false;
      }
      
      // Only consider assets that are in view frustum
      if (!inViewFrustum) {
        visibilityScore = 0.0f; // Completely exclude assets outside view frustum
        std::cout << "Asset " << i << ": pos(" << assetPos.x << ", " << assetPos.z 
                  << "), OUTSIDE VIEW FRUSTUM (h_angle=" << (horizontalAngle * 180.0f / M_PI) 
                  << "°, v_angle=" << (verticalAngle * 180.0f / M_PI) << "°)" << std::endl;
      } else {
        // Score based on:
        // 1. Distance: closer is better (but not too close)
        // 2. Angle: closer to camera direction is better
        // 3. Position: prefer assets closer to scene center
        
        float distanceScore = 1.0f / (1.0f + distance * 0.1f); // Closer = higher score
        float angleScore = 1.0f - (angleFromCenter / M_PI); // 0° = 1.0, 180° = 0.0
        float centerScore = 1.0f - (glm::length(assetPos) / m_groundSize); // Closer to center = higher score
        
        visibilityScore = distanceScore * 0.4f + angleScore * 0.4f + centerScore * 0.2f;
        
        std::cout << "Asset " << i << ": pos(" << assetPos.x << ", " << assetPos.z 
                  << "), distance=" << distance << ", angle=" << (angleFromCenter * 180.0f / M_PI) 
                  << "°, IN VIEW FRUSTUM, score=" << visibilityScore << std::endl;
      }
    } else {
      // Fallback: use random score for assets without circles
      visibilityScore = 0.5f;
    }
    
    assetScores.push_back({i, visibilityScore});
  }
  
  // Filter out assets with zero score (outside view frustum)
  std::vector<std::pair<int, float>> validAssets;
  for (const auto& score : assetScores) {
    if (score.second > 0.0f) {
      validAssets.push_back(score);
    }
  }
  
  if (validAssets.empty()) {
    std::cout << "WARNING: No assets found within camera view frustum!" << std::endl;
    std::cout << "Falling back to selecting from all assets..." << std::endl;
    validAssets = assetScores; // Use all assets as fallback
  }
  
  // Sort by visibility score (highest first)
  std::sort(validAssets.begin(), validAssets.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.second > b.second;
            });
  
  // Select from top 3 most visible assets (or all if less than 3)
  int candidatesCount = std::min(3, (int)validAssets.size());
  std::uniform_int_distribution<> dist(0, candidatesCount - 1);
  int selectedCandidate = dist(gen);
  
  m_targetAssetIndex = validAssets[selectedCandidate].first;
  
  std::cout << "Top " << candidatesCount << " visible assets (within view frustum):" << std::endl;
  for (int i = 0; i < candidatesCount; ++i) {
    std::cout << "  " << (i + 1) << ". Asset " << validAssets[i].first 
              << " (score: " << std::fixed << std::setprecision(3) << validAssets[i].second << ")";
    if (i == selectedCandidate) {
      std::cout << " ← SELECTED";
    }
    std::cout << std::endl;
  }
  
  std::cout << "Selected target asset index: " << m_targetAssetIndex << std::endl;
  
  
  // Initialize asset visibility (all visible by default)
  m_assetVisibility.clear();
  m_assetVisibility.resize(m_assets.size(), true);
  
  std::cout << "Target asset will be used for comparison rendering" << std::endl;
}

void Application::hideTargetAsset() {
  if (m_targetAssetIndex >= 0 && m_targetAssetIndex < m_assetVisibility.size()) {
    // Only reset render buffers if asset was previously visible
    bool wasVisible = m_assetVisibility[m_targetAssetIndex];
    
    m_assetVisibility[m_targetAssetIndex] = false;
    std::cout << "Hidden target asset (index " << m_targetAssetIndex << ")" << std::endl;
    
    // Debug: Print current visibility state
    std::cout << "Current asset visibility state: ";
    for (size_t i = 0; i < m_assetVisibility.size(); ++i) {
      std::cout << "[" << i << "]=" << (m_assetVisibility[i] ? "visible" : "hidden") << " ";
    }
    std::cout << std::endl;
    
    
    // Find the instance belonging to the target asset
    // Each asset creates one instance, so the instance index should match the asset index
    // (assuming ground is at index 0 and assets start from index 1)
    size_t instanceIndex = m_targetAssetIndex + 1; // +1 to account for ground instance
    
    if (instanceIndex < m_instances.size()) {
      // Mark the instance as invisible by setting visibility mask to 0
      // We need to rebuild the acceleration structure with updated visibility
      std::cout << "Hiding instance " << instanceIndex << " for target asset " << m_targetAssetIndex << std::endl;
      
      // Store the original visibility mask for restoration
      // For now, we'll rebuild the acceleration structure which will exclude the hidden instance
    }
  }
}

void Application::showTargetAsset() {
  if (m_targetAssetIndex >= 0 && m_targetAssetIndex < m_assetVisibility.size()) {
    // Only reset render buffers if asset was previously hidden
    bool wasHidden = !m_assetVisibility[m_targetAssetIndex];
    
    m_assetVisibility[m_targetAssetIndex] = true;
    std::cout << "Shown target asset (index " << m_targetAssetIndex << ")" << std::endl;
    
    // Debug: Print current visibility state
    std::cout << "Current asset visibility state: ";
    for (size_t i = 0; i < m_assetVisibility.size(); ++i) {
      std::cout << "[" << i << "]=" << (m_assetVisibility[i] ? "visible" : "hidden") << " ";
    }
    std::cout << std::endl;
    
  }
}

void Application::resetRenderBuffers() {
  std::cout << "Resetting render buffers..." << std::endl;
  
  // Reset iteration counter to start fresh rendering
  m_launchParameters.iteration = 0;
  
  // Clear device buffers
  const size_t numElements = size_t(m_resolution.x) * size_t(m_resolution.y);
  const size_t bufferSize = numElements * sizeof(float4);
  
  if (m_launchParameters.bufferAccum != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferAccum), 0, bufferSize));
  }
  if (m_launchParameters.bufferAlbedo != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferAlbedo), 0, bufferSize));
  }
  if (m_launchParameters.bufferDepth != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferDepth), 0, bufferSize));
  }
  if (m_launchParameters.bufferNormal != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferNormal), 0, bufferSize));
  }
  if (m_launchParameters.bufferRoughness != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferRoughness), 0, bufferSize));
  }
  if (m_launchParameters.bufferMetallic != 0) {
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_launchParameters.bufferMetallic), 0, bufferSize));
  }
  
  std::cout << "Render buffers reset complete" << std::endl;
}


void Application::renderWithCurrentEnvironment(const std::string& baseFilename, const std::string& hdriPath) {
  std::cout << "Rendering and saving images with base filename: " << baseFilename << std::endl;
  
  // Create output directory for this set of images
  std::string outputDir = "comparison_output/" + baseFilename;
  try {
    std::filesystem::create_directories(outputDir);
    std::cout << "Created output directory: " << outputDir << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Error creating directory " << outputDir << ": " << e.what() << std::endl;
    return;
  }
  
  // Copy HDRI file to the output directory
  if (!hdriPath.empty()) {
    try {
      std::filesystem::path sourceHdri(hdriPath);
      std::filesystem::path destHdri = outputDir + "/" + sourceHdri.filename().string();
      std::filesystem::copy_file(sourceHdri, destHdri);
      std::cout << "Copied HDRI file: " << sourceHdri.filename() << " to " << outputDir << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Error copying HDRI file: " << e.what() << std::endl;
    }
  }
  
  // Save multi-channel render output with directory prefix
  std::string fullBaseFilename = outputDir + "/" + baseFilename;
  saveRenderChannels(fullBaseFilename);
  
  std::cout << "Saved render channels for: " << baseFilename << " in directory: " << outputDir << std::endl;
}
