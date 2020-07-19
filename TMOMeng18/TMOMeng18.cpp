/* --------------------------------------------------------------------------- *
 * TMOMeng18.cpp: implementation of the TMOMeng18 class.   *
 * --------------------------------------------------------------------------- */

#include "TMOMeng18.h"
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include <ios>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/kernels/summary_interface.h>
#include <tensorflow/core/summary/summary_file_writer.h>
#include <tensorflow/cc/framework/ops.h>

using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace ops;


/* --------------------------------------------------------------------------- *
 * Constructor serves for describing a technique and input parameters          *
 * --------------------------------------------------------------------------- */
TMOMeng18::TMOMeng18() {
    SetName(L"Meng18");
    SetDescription(L"Convert source from color to grayscale or invertible.");
    mode.SetName(L"m");
    mode.SetDescription(
            L"1 - use loaded model from path, 2 - Build model and use, 3 - Build, train and save model");
    mode.SetDefault(1);
    mode = 1;
    mode.SetRange(1, 3);
    this->Register(mode);
    modelDirPath.SetName(L"g");
    modelDirPath.SetDescription(
            L"Path to directory with tensorflow graph( default ../TMOMeng18/graphs/InvertibleGrayscale)");
    modelDirPath.SetDefault("../TMOMeng18/graphs/InvertibleGrayscale");
    modelDirPath = "../TMOMeng18/graphs/InvertibleGrayscale";
    this->Register(modelDirPath);
    dataDirPath.SetName(L"d");
    dataDirPath.SetDescription(
            L"Path to directory with training data( default ../TMOMeng18/data/train)");
    dataDirPath.SetDefault("../TMOMeng18/data/test");
    dataDirPath = "../TMOMeng18/data/test";
    this->Register(dataDirPath);
    direction.SetName(L"c");
    direction.SetDescription(
            L"Convert to grayscale or to color. value:color,gray (default gray)");
    direction.SetDefault("gray");
    direction = "gray";
    this->Register(direction);
}

TMOMeng18::~TMOMeng18() {
}

/* --------------------------------------------------------------------------- *
 * This overloaded function is an implementation of your tone mapping operator *
 * --------------------------------------------------------------------------- */
int TMOMeng18::Transform() {
    if (mode.GetInt() == 1) {
        model = new InvertibleGrayscale(iWidth, iHeight, 3);
        auto status = model->LoadGraphFromFile(modelDirPath.GetString());
        if (status != Status::OK()) {
            return -1;
        }
    } else if (mode.GetInt() == 2) {
        model = new InvertibleGrayscale(iWidth, iHeight, 3);
        auto status = model->Build();
        if (status != Status::OK()) {
            wprintf(L"error: %s", status.error_message().c_str());
            return -1;
        }
    } else {
        model = new InvertibleGrayscale(256,256, 3);
        auto status = model->BuildAndTrain(dataDirPath.GetString());
        if (status != Status::OK()) {
            wprintf(L"error: %s", status.error_message().c_str());
            return -1;
        }
        status = model->SaveTrainedGraph(modelDirPath.GetString());
        if (status != Status::OK()) {
            wprintf(L"error: %s", status.error_message().c_str());
            return -1;
        }
    }
    auto status = Predict();
    if (status != Status::OK()) {
        return -1;
    }
    return 0;
}

Status TMOMeng18::Predict() {
    int depth =  (direction.GetString() == "color") ? 1 : 3;
    Tensor *t = new Tensor(DT_FLOAT, {1, this->iHeight, this->iWidth, depth});
    auto input_tensor_mapped = t->tensor<float, 4>();
    for (int y = 0; y < this->iWidth; y++) {
        const double *source_row = this->pSrc->GetData() + (y * this->iWidth * depth);
        for (int x = 0; x < this->iHeight; x++) {
            const double *source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; c++) {
                const double *source_value = source_pixel + c;
                input_tensor_mapped(0, y, x, c) = (float) *source_value;
            }
        }
    }
    std::vector<tensorflow::Tensor> outputs;
    int directionInt = (direction.GetString() == "color") ? 1 : 0;
    auto status = model->Predict(*t, outputs,directionInt);
    depth = 3;
    auto output_tensor_mapped = outputs[0].tensor<float, 4>();
    for (int y = 0; y < this->iWidth; y++) {
        for (int x = 0; x < this->iHeight; x++) {
            for (int c = 0; c < depth; c++) {
                this->pDst->GetData()[this->iWidth * y * depth + x * depth + c] =
                        (double) output_tensor_mapped(0, y, x, c);
            }
        }
    }
    return status;
}
