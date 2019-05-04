#pragma once
#include <string>

void StartProgressReporter();

void ReportProgress(std::string CurrentText, int CurrentValue, std::string OverallText, int OverallValue, bool IgnoreReportInterval = false);

void SetTotalTexels(size_t InTotalTexels);

void ReportCurrentFinishedTexels(size_t CurrentFinishedTexels);

void ReportProgressTextureMapping(
	int CurrentBlock,
	int TotalBlock,
	float elapsedTime,
	double LastRayTracingPerformance,
	double accumulatedGPUTime,
	double OverallRayTracingPerformance
);
