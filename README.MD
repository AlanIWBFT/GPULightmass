Luoshuang's GPULightmass for UE4
=============

Build
--------
Requires CUDA SDK v10.1 (the CUDA SDK version can be easily downgraded), tbb and embree3. When compiling with VS2017, install v140 (VS2015) toolset for the CUDA SDK to function properly.

embree3 is used as static library. Compile your own static lib version of embree3 by grabbing from https://github.com/embree/embree.

To create a usable distribution, you will need GPULightmass's UE4 integration: https://github.com/AlanIWBFT/UnrealEngine/tree/GPULightmass-4.22.0-release. Note that the integration code is covered by UE4 EULA. You will get a 404 if you haven't linked your GitHub account with Epic account and accepted the EULA. There are also post-build commands in `GPULightmassKernel.vcxproj` which copy the files into the engine folders. Modify them according to your engine location.
