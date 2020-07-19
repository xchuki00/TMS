//
// Created by patrik on 14.05.20.
//

#ifndef CTG_VGG19_H
#define CTG_VGG19_H

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/client/client_session.h>

using namespace std;
using namespace tensorflow;

class VGG19 {
public:
    typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;
    SavedModelBundle bundle;
    Output Vgginput;

    VGG19();

    VGG19(Scope scope);

    tensorflow::Status test(std::vector<Tensor> data);

    Output getVgg19(Scope scope, Input input,bool addAll);

};


#endif //CTG_VGG19_H
