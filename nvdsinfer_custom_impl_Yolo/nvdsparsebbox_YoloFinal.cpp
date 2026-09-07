// Standard end-to-end detector rows: x1, y1, x2, y2, confidence, class.
// Model-space decoding only. nvinfer owns frame projection and metadata.
#include "nvdsinfer_custom_impl.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" bool NvDsInferParseYoloFinal(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    objects.clear();
    auto invalid = []() {
        std::cerr << "NvDsInferParseYoloFinal: expected FLOAT [N,6] final detections "
                     "and valid class/threshold metadata" << std::endl;
        return false;
    };
    if (layers.size() != 1 || !network.width || !network.height ||
        !params.numClassesConfigured ||
        params.perClassPreclusterThreshold.size() < params.numClassesConfigured)
        return invalid();
    const auto& layer = layers.front();
    if (layer.dataType != FLOAT || layer.inferDims.numDims != 2 ||
        layer.inferDims.d[1] != 6 || layer.inferDims.d[0] < 0)
        return invalid();
    const size_t count = static_cast<size_t>(layer.inferDims.d[0]);
    if (count * 6 != layer.inferDims.numElements || (count && !layer.buffer))
        return invalid();
    for (unsigned c = 0; c < params.numClassesConfigured; ++c)
        if (!std::isfinite(params.perClassPreclusterThreshold[c])) return invalid();
    const float* data = static_cast<const float*>(layer.buffer);
    std::vector<NvDsInferObjectDetectionInfo> decoded;
    decoded.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const float* row = data + i * 6;
        for (unsigned j = 0; j < 6; ++j)
            if (!std::isfinite(row[j])) return invalid();
        if (row[5] < 0 || row[5] >= params.numClassesConfigured ||
            std::floor(row[5]) != row[5] || row[4] < 0 || row[4] > 1 ||
            row[2] < row[0] || row[3] < row[1]) return invalid();
        const unsigned cls = static_cast<unsigned>(row[5]);
        // The portable framework's confidence boundary is inclusive.
        if (row[4] < params.perClassPreclusterThreshold[cls]) continue;
        NvDsInferObjectDetectionInfo object{};
        object.left = row[0]; object.top = row[1];
        object.width = row[2] - row[0]; object.height = row[3] - row[1];
        if (!std::isfinite(object.width) || !std::isfinite(object.height)) return invalid();
        object.detectionConfidence = row[4]; object.classId = cls;
        decoded.push_back(object);
    }
    objects.swap(decoded);
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloFinal);
