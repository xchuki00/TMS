//
// Created by patrik on 18.05.20.
//

#ifndef CTG_IMAGELOADER_H
#define CTG_IMAGELOADER_H

#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include <cmath>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class ImageLoader {
public:
    Scope i_root; //graph for loading images into tensors
    const int imageWidth; //assuming quare picture
    const int imageHeight; //assuming quare picture
    const int image_channels; //RGB
    Output file_name_var;
    Output image_tensor_var;

    ImageLoader(int width,int height, int channels) : i_root(Scope::NewRootScope()),
                                          imageWidth(width),imageHeight(height), image_channels(channels) {}

    Status CreateGraphForImage(bool unstack);

    Status ReadTensorFromImageFile(string &file_name, Tensor &outTensor);

    Status ReadFileTensors(string &folder_name, vector<pair<Tensor, string>> &file_tensors);

    Status ReadBatches(string folder_name, int batch_size,
                       vector<Tensor> &image_batches);
};


#endif //CTG_IMAGELOADER_H
