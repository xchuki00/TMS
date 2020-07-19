#include "VGG19.h"
#include <iostream>
#include <fstream>

VGG19::VGG19() {};

VGG19::VGG19(Scope scope) {
    const std::string graph_fn = "../TMOMeng18/graphs/vgg19";
    TF_CHECK_OK(LoadSavedModel(SessionOptions(), RunOptions(), graph_fn, {"serve"}, &bundle));
}

tensorflow::Status VGG19::test(std::vector<Tensor> data) {
    const std::string graph_fn = "../TMOMeng18/graphs/vgg19";
    SavedModelBundle bundle;
    Status load_status = LoadSavedModel(
            SessionOptions(), RunOptions(), graph_fn, {"serve"}, &bundle);
    if (!load_status.ok()) {
        std::cout << "Error loading model: " << load_status << std::endl;
        return load_status;
    }

    tensor_dict feed_dict2 = {
            {"serving_default_input_1", data[0]}
    };
////        cout<<bundle.meta_graph_def.<<endl;
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(bundle.GetSession()->Run(feed_dict2, {"StatefulPartitionedCall"}, {}, &outputs));

    std::cout << "input           " << data[0].DebugString() << std::endl;
    std::cout << "output          " << outputs[0].DebugString() << std::endl;
    cout << "value: " << outputs[0].dim_size(-1) << endl;

    return tensorflow::Status::OK();
}

Output VGG19::getVgg19(Scope scope, Input input, bool addAll) {
    Node *outputNode;
    if (addAll) {
        TF_CHECK_OK(scope.graph()->AddFunctionLibrary(bundle.meta_graph_def.graph_def().library()));
    }
//    bundle.meta_graph_def.CopyFrom(bundle.meta_graph_def);
    for (auto node: bundle.meta_graph_def.graph_def().node()) {
        std::size_t found = node.name().find("StatefulPartitionedCall");
        if (node.name() == "StatefulPartitionedCall") {
            cout << input.node()->name() << endl;
            node.set_input(0, input.node()->name());
            node.set_name(input.node()->name() + "/" + node.name());
            Status addNodeStatus;
            outputNode = scope.graph()->AddNode(node, &addNodeStatus);
            scope.graph()->AddEdge(input.node(), 0, outputNode, 0);
            TF_CHECK_OK(addNodeStatus);
        } else if (node.name() != "serving_default_input_1" && found == std::string::npos && addAll) {
            Status addNodeStatus;
            scope.graph()->AddNode(node, &addNodeStatus);
            TF_CHECK_OK(addNodeStatus);
        }
    }
    GraphDef g;
    scope.ToGraphDef(&g);

    ofstream myfile;
    myfile.open("output.cpp");
    myfile << g.DebugString();
    myfile.close();
//
    return Output(outputNode);
}