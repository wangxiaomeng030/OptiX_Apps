/* 
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef APPLICATION_H
#define APPLICATION_H

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <windows.h>
#endif

// IMGUI
#define IMGUI_DEFINE_MATH_OPERATORS 1
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// GLEW
#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

// CUDA Runtime API version. Needs to be included after OpenGL headers!
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

// GLM
// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

// FASTGLTF
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include "Options.h"
#include "Logger.h"

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <stdint.h>

#include "Arena.h"

#include "DeviceBuffer.h"

#include "Camera.h"
#include "Light.h"
#include "Mesh.h"
#include "Node.h"
#include "Skin.h"

#include "Trackball.h"
#include "SceneExtent.h"

#include "Picture.h"
#include "Texture.h"

#include "cuda/config.h"
#include "cuda/launch_parameters.h"
#include "cuda/geometry_data.h"
#include "cuda/material_data.h"


#define APP_EXIT_SUCCESS          0

#define APP_ERROR_UNKNOWN        -1
#define APP_ERROR_CREATE_WINDOW  -2
#define APP_ERROR_GLFW_INIT      -3
#define APP_ERROR_GLEW_INIT      -4
#define APP_ERROR_APP_INIT       -5
#define APP_ERROR_EXCEPTION      -6

// The maximum number of values inside the m_benchmarkValues vector over which a running average is built.
#define SIZE_BENCHMARK_VALUES 100

// OpenGL-CUDA interop modes of the renderer.
// Change with command line option --interop (-i) <0|1|2|3> (default is 0)
enum InteropMode
{
  INTEROP_OFF = 0, // Device-to-host copy from native CUDA buffer to m_bufferHost and then host-to-device copy to m_hdrTexture with glTexSubImage2D. 
                   // Slowest method but the default because it works with any OpenGL implementation supporting RGBA32F textures.
                   // No need to have OpenGL running on the selected CUDA device, so also works on CUDA devices in TCC mode and separate display device.
  INTEROP_PBO = 1, // Direct rendering into mapped linear PBO and device-to-device copy to m_hdrTexture with glTexSubImage2D.
  INTEROP_TEX = 2, // Device-to-device copy from native CUDA buffer to m_hdrTexture mapped as CUDA texture array.
  INTEROP_IMG = 3  // Direct rendering into the mapped CUDA texture array image surface object of m_hdrTexture.
};

enum MiscConstants
{
  MAX_TRACE_DEPTH = 2
};


enum GuiState
{
  GUI_STATE_NONE,
  GUI_STATE_ORBIT,
  GUI_STATE_PAN,
  GUI_STATE_DOLLY,
  GUI_STATE_FOCUS
};


enum ModuleIdentifier
{
  MODULE_ID_RAYGENERATION,
  MODULE_ID_EXCEPTION,
  MODULE_ID_MISS,
  MODULE_ID_HIT,
  MODULE_ID_LIGHT_SAMPLE,

  NUM_MODULE_IDENTIFIERS
  // built-in modules don't have and ID (spheres, curves)
};


enum ProgramGroupId
{
  PGID_RAYGENERATION,
  
  PGID_EXCEPTION,
  
  PGID_MISS_RADIANCE,
  PGID_MISS_SHADOW,
  
  // Hit records for triangles:
  PGID_HIT_RADIANCE_TRIANGLES,
  PGID_HIT_SHADOW_TRIANGLES,

  // Hit records for spheres:
  PGID_HIT_RADIANCE_SPHERES,
  PGID_HIT_SHADOW_SPHERES,

  // Direct Callables (light sampling)
  // Area lights: 
  PGID_LIGHT_ENV_CONSTANT,
  PGID_LIGHT_ENV_SPHERE,
  // Singular lights:
  PGID_LIGHT_POINT,
  PGID_LIGHT_SPOT,
  PGID_LIGHT_DIRECTIONAL,

  // Number of all hardcoded program group entries. 
  NUM_PROGRAM_GROUP_IDS
};


namespace dev
{

  struct Instance
  {
    glm::mat4x4 transform;
    int         indexDeviceMesh; // Index into m_deviceMeshes.
  };
} // namespace dev;


class Application
{
public:

  enum BenchmarkMode : int
  {
    OFF,
    FPS,                // frames / second, render and display
    SAMPLES_PER_SECOND  // samples / second, pure raytracing performance
  };

  Application(GLFWwindow* window, Options const& options);
  ~Application();

  void reshape(int width, int height);
  
  bool render(); // Returns true if a new texture image is available for display.
  void update();
  void display();

  void guiNewFrame();
  void guiWindow();
  void guiEventHandler();
  void guiRender();


  int  getBenchmarkMode() const;
  void setBenchmarkValue(const float value); 

private:

  void initOpenGL();

  void checkInfoLog(const char *msg, GLuint object);
  void initGLSL();

  void updateProjectionMatrix();
  void updateVertexAttributes();
  float2 getPickingCoordinate(const int x, const int y);

  void getSystemInformation();
  void initCUDA();

  OptixResult initOptiXFunctionTable();
  void initOptiX();

  void addAssetsToScene();
  void createMultipleGLBInstances();
  void createGLBInstancesForAsset(size_t assetIndex, fastgltf::Asset& asset);
  int mapToGlobalMaterialIndex(size_t assetIndex, int localMaterialIndex);
  void processPrimitiveAttributes(const fastgltf::Primitive& primitive, 
                                fastgltf::Asset& asset, 
                                dev::HostPrimitive& hostPrim, 
                                bool fullMode);
  void loadGLTF(const std::filesystem::path& path);
  void createRandomScene();
  void createGround();
  void addProceduralPrimitives();
  fastgltf::Extensions getGLTFExtensions();
  fastgltf::Asset loadGLTFIntoAsset(const std::filesystem::path& path);
  fastgltf::Asset loadGLTFIntoAsset(const std::string& glb_path);
  void loadHDREnvironmentLight();
  void generateRandomLayout();
  template<typename T>
  void printExtensions(const char* info, const T& extensions);
  void initRenderer();

  std::vector<char> readData(std::string const& filename);

  // ArenaAllocator versions of cudaMalloc and cudaFree. Attention: These are asynchronous to other cuda functions!
  CUdeviceptr memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage);
  void memFree(const CUdeviceptr ptr);

  void updateCamera();
  void updateBuffers();
  
  // Denoiser related functions
  void initDenoiser();
  void cleanupDenoiser();
  void invokeDenoiser();
  void setDenoiserImages();

  void addImage(const int32_t width,
                const int32_t height,
                const int32_t bitsPerComponent,
                const int32_t numComponents,
                const void*   data);

  void addSampler(cudaTextureAddressMode address_s,
                  cudaTextureAddressMode address_t,
                  cudaTextureFilterMode  filter,
                  const size_t           image_idx,
                  const int              sRGB);

  void cleanup(); // Complete destruction of all resources, called in ~Application().

  void buildDeviceMeshAccel(const int indexDeviceMesh, const bool rebuild);
  void buildDeviceMeshAccels(const bool rebuild);
  void buildInstanceAccel(const bool rebuild);

  void initPipeline();
  void initSBT();


  void initLaunchParameters();
  void updateLaunchParameters();

  void initNodes();
  size_t calculateTotalNodes();
  void initImages();
  void initTextures(size_t groundImageCount, size_t groundSamplerCount);
  void initMaterials(size_t groundSamplerCount);
  void initMeshes();



  void initTrackball();


  void initSheenLUT();

  LightDefinition createSphericalEnvironmentLight();
  LightDefinition createSphericalEnvironmentLightFromExistingTexture();
  
  void updateBufferHost();
  bool screenshot(const bool tonemap);

  /// Init a device primitive from a host primitive. Also sets the primitive type.
  void createDevicePrimitive(dev::DevicePrimitive& devicePrim, const dev::HostPrimitive& hostPrim, const int skin);

  /// Create all device primitives for a given mesh.
  /// @param deviceMesh OUT
  /// @param hostKey    Index into the host meshes
  void createDeviceMesh(dev::DeviceMesh& deviceMesh, const dev::KeyTuple hostKey);

  void updateFonts();

  /// flags for accelBuild
  unsigned int getBuildFlags() const;

  // Move camera (on keyboard input)
  void cameraTranslate(float dx, float dy, float dz);
  
  // Load random PBR material from the PBR materials folder
  bool loadRandomPBRMaterial(MaterialData& material);
  
  // Analyze HDR environment brightness for material compatibility
  
  // Check material compatibility with current HDR environment
  bool checkMaterialCompatibility(const MaterialData& material, float& recommendedBrightness);
  
  
  // Select appropriate ground material based on HDR environment
  void selectAppropriateGroundMaterial(MaterialData& groundMaterial);
  
  // Multi-channel rendering and saving
  void saveRenderChannels(const std::string& baseFilename);
  void saveImageChannel(const std::string& filename, const float4* data, int width, int height, const std::string& channelName);
  
  // Analyze HDR brightness for debugging purposes
  float analyzeHDRBrightness(const std::string& hdrPath);
  
  // Analyze HDR directional strength for better shadow generation
  float analyzeHDRDirectionality(const std::string& hdrPath);
  
  // Load stronger directional HDR for HDR2 in comparison mode
  std::vector<std::string> loadStrongerDirectionalHDR(int count);
  
  // Asset comparison rendering functionality
  void renderAssetComparison();
  std::vector<std::string> loadRandomHDRIEnvironments(int count = 2);
  bool validateHDRIFile(const std::string& hdrPath);
  void selectTargetAsset();
  void hideTargetAsset();
  void showTargetAsset();
  void resetRenderBuffers();
  void renderWithCurrentEnvironment(const std::string& baseFilename, const std::string& hdriPath = "");

private:
  GLFWwindow* m_window;
  Options const& m_options;  // Reference to command line options

  // Application command line parameters.
  std::filesystem::path m_pathAsset;
  std::string m_glb_path = "/data/codes/optix_all/assets_store_new/objaverse_opp_36k";
  std::string m_path_pbr = "/data/codes/optix_all/assets_store/pbr";
  std::string m_path_hdri = "/data/codes/optix_all/assets_store/hdri";

  int         m_width;      // Client window width.
  int         m_height;     // Client window height.
  int2        m_resolution; // Render resolution, independent of the client window size (m_width, m_height).
  int         m_interop;    // 0 == off, 1 == pbo, 2 = copy to cudaArray, 3 = surface read/write
  bool        m_punctual;   // Support for KHR_lights_punctual, default true.
  int         m_missID;     // 0 = null, 1 = constant, 2 = spherical environment (default).
  std::string m_pathEnv;    // Command line option --env (-e) <path.hdr> sets m_pathEnv.


  std::vector<fastgltf::Asset> m_assets; // The glTF assets when the loading succeeded.
  
  // GUI values for the environment lights.
  float m_colorEnv[3]    = { 1.0f, 1.0f, 1.0f };
  float m_intensityEnv   = 1.0f;
  float m_rotationEnv[3] = { 0.0f, 0.0f, 0.0f }; // The Euler rotation angles for the spherical environment light.


  // OpenGL variables:
  GLuint m_pbo = 0;
  GLuint m_hdrTexture = 0;

  float4* m_bufferHost = nullptr;
  
  // Multi-channel rendering buffers
  float4* m_bufferAlbedo = nullptr;    // Albedo channel
  float4* m_bufferDepth = nullptr;     // Depth channel  
  float4* m_bufferNormal = nullptr;    // Normal channel
  float4* m_bufferRoughness = nullptr; // Roughness channel
  float4* m_bufferMetallic = nullptr;   // Metallic channel
  float4* m_bufferMask = nullptr;      // Mask channel

  // Denoiser related variables
  OptixDenoiser m_denoiser = nullptr;
  CUdeviceptr m_d_stateDenoiser = 0;
  CUdeviceptr m_d_scratchDenoiser = 0;
  CUdeviceptr m_d_denoisedBuffer = 0;
  OptixDenoiserSizes m_sizesDenoiser = {};
  size_t m_scratchSizeInBytes = 0;
  OptixDenoiserParams m_paramsDenoiser = {};
  OptixImage2D m_inputImage[3] = {}; // RGB, Albedo, Normal
  OptixImage2D m_outputImage = {};
  OptixDenoiserGuideLayer m_guideLayer = {};
  OptixDenoiserLayer m_layer = {};
  int m_numInputLayers = 1; // At least the beauty buffer is always used
  bool m_enableDenoiser = true; // Enable/disable denoiser

  int m_launches = 1; // The number of asynchronous launches per render() call. Can be set with command line option --launches (-l) <int>

  BenchmarkMode m_benchmarkMode = BenchmarkMode::OFF;

  int m_benchmarkEntries = 0;   // The current number of valid benchmark results inside the vector.
  int m_benchmarkCell    = 0;   // The next cell inside the m_benchmarkValues vector to be written to.
  std::vector<float> m_benchmarkValues;

  // Data used to determine if the active CUDA device is also running the OpenGL implementation, otherwise no OpenGL-CUDA interop is possible.
  CUuuid m_cudaDeviceUUID;

  // Tonemapper group:
  float  m_gamma;
  float3 m_colorBalance;
  float  m_whitePoint;
  float  m_burnHighlights;
  float  m_crushBlacks;
  float  m_saturation;
  float  m_brightness;

  GuiState m_guiState;
  
  bool m_isVisibleGUI; // Hide the GUI window completely with SPACE key.

  ImFont* m_font = nullptr;
  float   m_fontScale = 0.0f;
  GLuint  m_fontTexture = 0;

  const float CameraSpeedFraction = 0.005f; // Camera translation (keyboard's A/W/S/D/... events)

  float m_mouseSpeedRatio = 100.0f; // Adjusts how many pixels the mouse needs to travel for 1 unit change for panning and dollying.
  bool  m_isLockedGimbal  = true;   // Toggle the gimbal lock on the trackball. true keeps the up-vector intact, false allows rolling.
  
  float m_epsilonFactor = 100.0f; // Self-intersection avoidance factor, multiplied with SCENE_EPSILON_SCALE to get final value. Mask rendering uses independent epsilon=0.

  // Some renderer global settings.
  bool m_useDirectLighting   = true;  // Switch between explicit direct lighting and brute force path tracing.
                                      // Singular lights only work with direct lighting! 
  bool m_useAmbientOcclusion = true;  // Global illumination handles ambient occlusion automatically,
                                      // but many glTF models are low resolution with high resolution detail baked into the normal and occlusion textures which look better when applying the occlusion tetxure attenuation.
                                      // This affects only diffuse and glossy (metal) reflections inside the renderer and only environment lights.
  bool m_showEnvironment     = true;  // Toggle the display of the environment for primary camera rays, shows white instead.
  bool m_forceUnlit          = false; // Force renderer to handle all materials as unlit. Useful in case the scene is not modeled correctly for global illumination (like VirtualCity.gltf).

  // OpenGL resources used inside the VBO path.
  GLuint m_vboAttributes = 0;
  GLuint m_vboIndices    = 0;

  // GLSL shaders objects and program.
  GLuint m_glslVS = 0;
  GLuint m_glslFS = 0;
  GLuint m_glslProgram = 0;

  GLint m_locAttrPosition = -1;
  GLint m_locAttrTexCoord = -1;
  GLint m_locProjection   = -1;
    
  //std::vector<cudaDeviceProp> m_deviceProperties;

  // CUDA native types are prefixed with "cuda".
  CUdevice  m_cudaDevice  = 0; // The CUdevice handle of the CUDA device ordinal. (Usually identical to the ordinal number.)
  CUcontext m_cudaContext = nullptr;
  CUstream  m_cudaStream  = nullptr;

  // The handle for the registered OpenGL PBO when using interop.
  CUgraphicsResource m_cudaGraphicsResource = nullptr;
  
  // Radius of the spheres for the glTF points.
  // Some datasets are tricky (huge bbox but most of the points are in a small
  // subvolume -> need to tweak the radius).
  // Could be a gui slider too.
  // Makes me think of yet another widget: clipping plane(s) to use with dense CT datasets.
  float              m_sphereRadiusFraction{ 0.005f };

  // All others are OptiX types.
  OptixFunctionTable m_api;
  OptixDeviceContext m_optixContext = nullptr;
  unsigned int m_visibilityMask = 255; // Instance Visibility Mask default. Queried and set in initOptiX().

  Logger m_logger;

  cuda::ArenaAllocator* m_allocator = nullptr;
  size_t m_sizeArena = 128; // Default to 128 MiB Arenas when nothing is specified inside the system description.

  Picture* m_picSheenLUT = nullptr; // Loads the "sheen_lut.hdr" image in initSheenLUT().
  Texture* m_texSheenLUT = nullptr; // Lookup table for the sheen sampling weight estimation. 2D float texture with lookup sheenWeight == lut(dot(V, N), sheenRoughness);

  dev::Camera m_camera; // Single default camera
  std::vector<dev::Instance>       m_instances;
  std::vector<dev::HostMesh>       m_hostMeshes;
  
  // Random layout data for object placement
  struct RandomCircle
  {
    glm::vec3 center;  // Center position on ground (y = 0)
    float radius;      // Radius of the circle
  };
  std::vector<RandomCircle> m_randomCircles;
  float m_groundSize; // Half-size of the ground plane (ground goes from -m_groundSize to +m_groundSize)
  
  // Asset comparison functionality
  int m_targetAssetIndex = -1; // Index of the selected target asset (-1 means none selected)
  std::vector<bool> m_assetVisibility; // Track visibility of each asset
  bool m_assetComparison = false; // Flag to indicate if we're in asset comparison mode
  int m_comparisonSamples = 64; // Number of samples for asset comparison rendering
  std::vector<MaterialData>        m_materialsOrg;  // The original material data inside the asset.
  std::vector<MaterialData>        m_materials;     // The material data changed by the GUI.
  std::vector<cudaArray_t>         m_images;        // Textures reference these images.
  std::vector<cudaTextureObject_t> m_samplers;      // Each texture has exactly one hardware sampler. 
                                                    // Sampler settings (wrap, filter) might be defined by glTF samplers (not sRGB though, see initTextures()).
  std::vector<dev::Node>           m_nodes;         // A shadow of the asset nodes holding just the local transformation (matrix or translation, rotation, scale).
  std::map<dev::KeyTuple, int> m_mapKeyTupleToDeviceMeshIndex; // For skinning and book-keeping.
  std::vector<dev::DeviceMesh> m_deviceMeshes;

  dev::SceneExtent m_sceneExtent;
  std::unordered_map<fastgltf::Primitive const*, HostBuffer const*> m_primitiveToHostBuffer; // rememeber the buffers to compute the extent after initMeshes()

  // Spherical HDR texture environment light.
  // Only used with command line option: --miss (-m) 2.
  // Supports drag-and-drop of *.hdr filenames then!
  Picture* m_picEnv = nullptr;
  Texture* m_texEnv = nullptr;

  std::vector<LightDefinition> m_lightDefinitions;
  CUdeviceptr m_d_lightDefinitions = 0;
  
  DeviceBuffer m_growInstances;

  OptixTraversableHandle m_ias = 0; // This is the root traversable handle of the current scene.
  DeviceBuffer m_growIas;     // The acceleration structure data of the root traversable IAS.
  DeviceBuffer m_growIasTemp; // The temporary memory required for rebuild or update of the IAS.

  CUdeviceptr m_d_iasAABB = 0; // Fixed allocation for the IAS AABB result.

  // Module and Pipeline data.
  OptixModuleCompileOptions   m_mco = {};
  OptixPipelineCompileOptions m_pco = {};
  OptixPipelineLinkOptions    m_plo = {};
  OptixProgramGroupOptions    m_pgo = {}; // This is a just placeholder.

  std::vector<std::string>       m_moduleFilenames;
  std::vector<OptixModule>       m_modules;
  OptixModule                    m_moduleBuiltinISSphere;
  std::vector<OptixProgramGroup> m_programGroups;

  OptixPipeline m_pipeline = 0;

  OptixShaderBindingTable m_sbt = {};

  LaunchParameters  m_launchParameters   = {};
  LaunchParameters* m_d_launchParameters = nullptr;

  size_t m_indexVariant  = 0; // The currently selected material variant.
  size_t m_indexMaterial = 0; // The currently selected material inside the GUI.

  // This is set to true in initCameras() when there was no camera inside the asset.
  // In that case a perspective camera had been added and that needs to be placed and centered
  // to the current scene's AABB which is only available later after the initScene() call.

  // All true to invoke necessary updateBuffers(), and updateLights().
  // Cameras track their isDirty state internally and updateCamera() clears it.
  bool m_isDirtyResolution = true;


  dev::Trackball m_trackball;

  // A vector which contains the asynchronously launched iteration values, filled inside the renderer() function.
  // Defining once here to not resize it on every render call but only when changing the launches value inside the GUI.
  std::vector<unsigned int> m_iterations;
};

#endif // APPLICATION_H

