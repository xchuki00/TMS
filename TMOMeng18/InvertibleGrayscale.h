
#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/framework/ops.h>
#include <cmath>
#include "VGG19.h"
#include "ImageLoader.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class InvertibleGrayscale {
private:
    Scope i_root; //graph for loading images into tensors
    const int imageWidth;
    const int imageHeight;

    const int imageChannels; //RGB
    //load image vars
    Output file_name_var;
    Output image_tensor_var;
    //training and validating the CNN
    Scope t_root; //graph
    unique_ptr<ClientSession> t_session;
    //CNN vars
    Output input_batch_var;
    Output input_gray_var;
    Output grayImage;
    Output newColorImage;
    Output loss_op1;
    Output loss_op2;
    SavedModelBundle LoadedBundle;
    bool isLoaded = false;

    //Network maps
    map<string, Output> m_vars;
    map<string, TensorShape> m_shapes;
    map<string, Output> m_assigns;
    //Loss variables
    vector<Operation> out_grads_op1;
    vector<Operation> out_grads_op2;
public:
    typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;
    InvertibleGrayscale(int width,int height, int channels) : i_root(Scope::NewRootScope()), t_root(Scope::NewRootScope()),
                                                  imageWidth(width),imageHeight(height), imageChannels(channels) {}

    Status CreateGraphForImage(bool unstack);

    Status ReadTensorFromImageFile(string &file_name, Tensor &outTensor);

    Status ReadFileTensors(string &folder_name, vector<pair<Tensor, string>> &file_tensors);

    Status ReadBatches(string &folder_name, int batch_size, vector<Tensor> &image_batches);

    Status LoadGraphFromFile(string path);

    Status BuildAndTrain(string base_folder, int batch_size = 8);

    Status Build();

    Status SaveTrainedGraph(string path);

    Input XavierInit(Scope scope, int in_chan, int out_chan, int filter_side = 0);

    Status CreateEncodeGraph();

    Status CreateDecodeGraph(Input input);

    Status CreateOptimizationGraph(float learning_rate,float epsilon);

    Status Initialize();

    Status TrainCNN(Tensor &image_batch, vector<Tensor> &results,int mode);

    Status Predict(Tensor &image, vector<Tensor> &result,int direction);

    Output AddConv2D(Scope scope, Input input, int k, int inputN, int outputN, int s, string name,bool train);

    Output firstLayer(Input input);

    Output lastLayer(Input input, int outputN, string name);

    Output residualLayer(Input input, int k, int inputN, int outputN, int s, string name,bool train);

    Output downLayer(Input input, int k, int inputN, int outputN, int s, string name);

    Output upLayer(Input input, Input down, int k, int inputN, int outputN, int s, int upsamplingCoenf, string name);

    Status Loss(Scope scope);

    Tensor getLuminace(Tensor &trueColorImage);

    Output TotalVariation(Scope scope, Input input, string name);

    Output MeanAxis(Scope scope, Input input, vector<int> axis, string name);
};

