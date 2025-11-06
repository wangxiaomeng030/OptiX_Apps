/* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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


#include "Options.h"
#include <GLFW/glfw3.h> // dpi 

#include <iostream>

static int toInt(float x)
{
  return static_cast<int>(x);
}

// public:

Options::Options()
  : m_widthClient(toInt(1280 * getViewportScale()))   // Fixed resolution for optimal speed
  , m_heightClient(toInt(704 * getViewportScale()))   // Fixed resolution for optimal speed
  , m_widthResolution(toInt(1280 * getViewportScale()))   // Fixed render resolution
  , m_heightResolution(toInt(704 * getViewportScale()))   // Fixed render resolution
  , m_launches(64)  // Increased for better quality, less noise, and clearer shadows
  , m_interop(0)
  , m_miss(2)
  , m_punctual(true)
  , m_sphereRadiusFraction(0.005f)
  , m_showEnvironment(false)  // Disable HDR environment background by default
  , m_assetComparison(false)  // Disable asset comparison by default
  , m_comparisonSamples(16)   // Default number of samples for asset comparison
  , m_hdr1Path("/data/codes/optix_all/assets_store/hdri/lakeside_sunrise_4k.hdr")  // Default HDR1 path
  , m_hdr2Path("/data/codes/optix_all/assets_store/hdri/rustig_koppie_puresky_4k.hdr")  // Default HDR2 path
{
}

//Options::~Options()
//{
//}

bool Options::parseCommandLine(int argc, char *argv[])
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);

    if (arg == "?" || arg == "help" || arg == "--help")
    {
      printUsage(std::string(argv[0])); // Application name.
      return false;
    }
    else if (arg == "-f" || arg == "--file")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_filename = std::filesystem::path(argv[++i]);
    }
    else if (arg == "-w" || arg == "--width")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_widthClient = atoi(argv[++i]);
    }
    else if (arg == "-h" || arg == "--height")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_heightClient = atoi(argv[++i]);
    }
    else if (arg == "-x" || arg == "--xres")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_widthResolution = atoi(argv[++i]);
    }
    else if (arg == "-y" || arg == "--yres")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_heightResolution = atoi(argv[++i]);
    }
    else if (arg == "-l" || arg == "--launches")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_launches = atoi(argv[++i]);
    }
    else if (arg == "-i" || arg == "--interop")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_interop = atoi(argv[++i]);
    }
    else if (arg == "-p" || arg == "--punctual")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_punctual = (atoi(argv[++i]) != 0);
    }
    else if (arg == "-m" || arg == "--miss")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_miss = atoi(argv[++i]);
    }
    else if (arg == "-e" || arg == "--env")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_environment = std::string(argv[++i]);
    }
    else if (arg == "-r" || arg == "--radius")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_sphereRadiusFraction = static_cast<float>(std::atof(argv[++i]));
    }
    else if (arg == "-s" || arg == "--show-env")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_showEnvironment = (atoi(argv[++i]) != 0);
    }
    else if (arg == "-c" || arg == "--compare")
    {
      m_assetComparison = true;
    }
    else if (arg == "--comparison-samples")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_comparisonSamples = atoi(argv[++i]);
    }
    else if (arg == "--hdr1")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_hdr1Path = std::string(argv[++i]);
    }
    else if (arg == "--hdr2")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_hdr2Path = std::string(argv[++i]);
    }
    else
    {
      std::cerr << "ERROR: Unknown option '" << arg << "'\n";
      printUsage(argv[0]);
      return false;
    }
  }

  // Validate all input values.
  bool isInvalidArgument = false;

  if (m_filename.empty())
  {
    std::cerr << "WARNING: No --file (-f) argument provided, starting with empty scene.\n(Use drag-and-drop of *.gltf or *.glb files to load asset.)\n";
  }
  
  if (m_widthClient <= 0)
  {
    std::cerr << "ERROR: Invalid --width (-w) argument (must be > 0).\n";
    isInvalidArgument = true;
  }
  
  if (m_heightClient <= 0)
  {
    std::cerr << "ERROR: Invalid --height (-h) argument (must be > 0).\n";
    isInvalidArgument = true;
  }

  if (m_widthResolution <= 0)
  {
    std::cerr << "ERROR: Invalid --xres (-x) argument (must be > 0).\n";
    isInvalidArgument = true;
  }
  
  if (m_heightResolution <= 0)
  {
    std::cerr << "ERROR: Invalid --yres (-y) argument (must be > 0).\n";
    isInvalidArgument = true;
  }

  if (m_launches < 1)
  {
    std::cerr << "WARNING: --launches (-l) needs to be greater than 0.\n";
    m_launches = 1; 
  }
  else if (MAX_LAUNCHES < m_launches)
  {
    std::cerr << "WARNING: --launches (-l) needs to be less than or equal to 1000.\n";
    m_launches = MAX_LAUNCHES;
  }

  if (m_interop < 0 || 3 < m_interop)
  {
    std::cerr << "ERROR: Invalid --interop (-i) argument (must be 0 to 3).\n";
    isInvalidArgument = true;
  }

  if (m_miss < 0 || 2 < m_miss)
  {
    std::cerr << "ERROR: Invalid --miss (-m) program argument (must be 0 to 2).\n";
    isInvalidArgument = true;
  }

  if (m_comparisonSamples < 1)
  {
    std::cerr << "ERROR: Invalid --comparison-samples argument (must be >= 1).\n";
    isInvalidArgument = true;
  }

  if (isInvalidArgument)
  {
    printUsage(argv[0]);
    return false;
  }

  return true;
}


std::filesystem::path Options::getFilename() const
{
  return m_filename;
}

int Options::getWidthClient() const
{
  return m_widthClient;
}

int Options::getHeightClient() const
{
  return m_heightClient;
}

int Options::getWidthResolution() const
{
  return m_widthResolution;
}

int Options::getHeightResolution() const
{
  return m_heightResolution;
}

int Options::getLaunches() const
{
  return m_launches;
}

int Options::getInterop() const
{
  return m_interop;
}

int Options::getMiss() const
{
  return m_miss;
}

std::string Options::getEnvironment() const
{
  return m_environment;
}

bool Options::getAssetComparison() const
{
  return m_assetComparison;
}

bool Options::getPunctual() const
{
  return m_punctual;
}

float Options::getSphereRadiusFraction() const
{
  return m_sphereRadiusFraction;
}

bool Options::getShowEnvironment() const
{
  return m_showEnvironment;
}

int Options::getComparisonSamples() const
{
  return m_comparisonSamples;
}

std::string Options::getHdr1Path() const
{
  return m_hdr1Path;
}

std::string Options::getHdr2Path() const
{
  return m_hdr2Path;
}

// private:

void Options::printUsage(std::string const& argv0)
{
  std::cerr << "\nUsage: " << argv0 << " [options]\n";
  std::cerr <<
    "Options:\n"
    "   ? | --help | help     Print this usage message and exit.\n"
    "  -f | --file <filename> Filename of a glTF model (required). (empty)\n"
    "  -w | --width <int>     Client window width.  (1280)\n"
    "  -h | --height <int>    Client window height. (720)\n"
    "  -x | --xres <int>      Render resolution width.  (1280)\n"
    "  -y | --yres <int>      Render resolution height. (720)\n"
    "  -l | --launches <int>  Number of launches per render call, range [1, 1000] (1)\n"
    "  -i | --interop <int>   OpenGL interop: 0 = off, 1 = pbo, 2 = array copy, 3 = surface write. (0)\n"
    "  -p | --punctual <int>  Select KHR_lights_punctual support: 0 = off. (1)\n"
    "  -m | --miss <0|1|2>    Select the miss shader: 0 = null, 1 = white, 2 = environment. (2))\n"
    "  -e | --env <filename>  Filename of a spherical HDR texture. (empty)\n"
    "  -r | --radius <fraction>  Fraction of the scene diameter. (0.005)\n"
    "  -s | --show-env <0|1>  Show environment background: 0 = off (white), 1 = on (HDRI). (0)\n"
    "  -c | --compare         Enable asset comparison rendering mode. (off)\n"
    "  --comparison-samples <int>  Number of samples for asset comparison rendering. (64)\n"
    "  --hdr1 <path>          Path to HDR1 environment file for comparison mode. (lakeside_sunrise_4k.hdr)\n"
    "  --hdr2 <path>          Path to HDR2 environment file for comparison mode. (rustig_koppie_puresky_4k.hdr)\n"
    "Viewport Interactions:\n"
    "  SPACE       Toggle GUI display.\n"
    "  P           Save image as tonemapped *.png.\n"
    "  H           Save image as linear *.hdr.\n"
    "  LMB         Orbit\n"
    "  MMB         Pan\n"
    "  RMB         Dolly\n"
    "  MouseWheel  Zoom\n"
    "  Ctrl+LMB    Select material.\n";
}

float Options::getViewportScale() const
{
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  float xscale, yscale;
  glfwGetMonitorContentScale(monitor, &xscale, &yscale);

  return xscale;
}
