
#include "InvertibleGrayscale.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

Input InvertibleGrayscale::XavierInit(Scope scope, int in_chan, int out_chan, int filter_side) {
    float std;
    Tensor t;
    std = sqrt(6.f / (filter_side * filter_side * (in_chan + out_chan)));
    Tensor ts(DT_INT64, {4});
    auto v = ts.vec<int64>();
    v(0) = filter_side;
    v(1) = filter_side;
    v(2) = in_chan;
    v(3) = out_chan;
    t = ts;
    auto rand = RandomUniform(scope, t, DT_FLOAT);
    return Multiply(scope, Sub(scope, rand, 0.5f), std * 2.f);
}

Status InvertibleGrayscale::CreateOptimizationGraph(float learning_rate, float epsilon) {
/*
# --------------------------------- solver definition ---------------------------------
                                           global_step = tf.Variable(0, name='global_step1', trainable=False)
    iters_per_epoch = np.floor_divide(num, batch_size)
    lr_decay = tf.train.polynomial_decay(learning_rate=learning_rate,
                                         global_step=global_step,
                                         decay_steps=iters_per_epoch * (n_epochs1 + n_epochs2),
                                         end_learning_rate=learning_rate / 100.0,
                                         power=0.9)

    with tf.name_scope('optimizer'):
    gen_vars = [var for var in tf.trainable_variables() if
        var.name.startswith("encode") or var.name.startswith("decode")]
    train_op1 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op1, var_list=gen_vars,
                                                                       global_step=global_step)
    train_op2 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op2, var_list=gen_vars,
                                                                       global_step=global_step)
*/
    input_gray_var = Placeholder(t_root.WithOpName("inputCorrect"), DT_FLOAT);
    Scope scope_loss = t_root.NewSubScope("Loss_scope");
    Loss(scope_loss);
    TF_CHECK_OK(scope_loss.status());
    vector<Output> weights_biases;
    for (pair<string, Output> i: m_vars) {
        std::size_t found = i.first.find("train");
        if (found != std::string::npos) {
            weights_biases.push_back(i.second);
        }
    }
    vector<Output> grad_outputs_op1, grad_outputs_op2;
    TF_CHECK_OK(AddSymbolicGradients(t_root, {loss_op1}, weights_biases, &grad_outputs_op1));
    TF_CHECK_OK(AddSymbolicGradients(t_root, {loss_op1}, weights_biases, &grad_outputs_op2));

    int index = 0;
    for (pair<string, Output> i: m_vars) {
        //Applying Adam
        std::size_t found = i.first.find("train");
        if (found != std::string::npos) {
            string s_index = to_string(index);
            auto m_var = Variable(t_root, m_shapes[i.first], DT_FLOAT);
            auto v_var = Variable(t_root, m_shapes[i.first], DT_FLOAT);
            m_assigns["m_assign" + s_index] = Assign(t_root, m_var, Input::Initializer(0.f, m_shapes[i.first]));
            m_assigns["v_assign" + s_index] = Assign(t_root, v_var, Input::Initializer(0.f, m_shapes[i.first]));

            auto adam_op1 = ApplyAdam(t_root, i.second, m_var, v_var, 0.9f, 0.9f, learning_rate, 0.9f, 0.9f, epsilon,
                                      {grad_outputs_op1[index]});
            auto adam_op2 = ApplyAdam(t_root, i.second, m_var, v_var, 0.9f, 0.9f, learning_rate, 0.9f, 0.9f, epsilon,
                                      {grad_outputs_op2[index]});
            out_grads_op1.push_back(adam_op1.operation);
            out_grads_op2.push_back(adam_op2.operation);
            index++;
        }

    }
    return t_root.status();

}

Output InvertibleGrayscale::AddConv2D(Scope scope, Input input, int k, int inputN, int outputN, int s, string name,
                                      bool train) {
    TensorShape sp({k, k, inputN, outputN});
    if (train) {
        name = name + "-train";
        m_vars["W" + name] = Variable(scope.WithOpName("W" + name), sp, DT_FLOAT);
        m_shapes["W" + name] = sp;
        m_assigns["W" + name + "_assign"] = Assign(scope.WithOpName("W_assign" + name), m_vars["W" + name],
                                                   XavierInit(scope, inputN, outputN, k));
        return Conv2D(scope.WithOpName("Conv" + name), input, m_vars["W" + name], {1, s, s, 1}, "SAME");
    } else {
        m_vars["W" + name] = Variable(scope.WithOpName("W" + name), sp, DT_FLOAT);
        m_shapes["W" + name] = sp;
        m_assigns["W" + name + "_assign"] = Assign(scope.WithOpName("W_assign" + name), m_vars["W" + name],
                                                   XavierInit(scope, inputN, outputN, k));
        return Conv2D(scope.WithOpName("Conv" + name), input, m_vars["W" + name], {1, s, s, 1}, "SAME");
//        return Conv2D(scope.WithOpName("Conv" + name), input, Variable(scope.WithOpName("W" + name), sp, DT_FLOAT),
//                      {1, s, s, 1}, "SAME");
    }
}

Output InvertibleGrayscale::firstLayer(Input input) {
    Scope scope = t_root.NewSubScope("firstLayer");
    auto conv = AddConv2D(scope, input, 3, 3, 64, 1, "0", false);
    return Relu(scope.WithOpName("Relu-0"), conv);
}

Output InvertibleGrayscale::lastLayer(Input input, int outputN, string name) {
    Scope scope = t_root.NewSubScope(name);
    auto conv = AddConv2D(scope, input, 3, 64, outputN, 1, name, false);
    return Tanh(scope.WithOpName("Tanh-" + name), conv);
}

Output InvertibleGrayscale::residualLayer(Input input, int k, int inputN, int outputN, int s, string name, bool train) {
    Scope scope = t_root.NewSubScope(name);
    auto inputSave = input;
    auto conv = AddConv2D(scope, input, k, inputN, outputN, s, name, train);
    auto relu = Relu(scope.WithOpName("Relu-" + name), conv);
    auto conv2 = AddConv2D(scope, relu, k, outputN, outputN, s, name + "-2", train);
    return Add(scope.WithOpName("Add-" + name), conv2, inputSave);
}

Output InvertibleGrayscale::downLayer(Input input, int k, int inputN, int outputN, int s, string name) {
    Scope scope = t_root.NewSubScope(name);
    auto conv = AddConv2D(scope, input, k, inputN, outputN, 2, name, true);
    auto conv2 = AddConv2D(scope, conv, k, outputN, outputN, s, name + "-2", true);
    return Relu(scope.WithOpName("Relu-" + name), conv2);
}

Output InvertibleGrayscale::upLayer(Input input, Input down, int k, int inputN, int outputN, int s, int upsamplingCoenf,
                                    string name) {
    Scope scope = t_root.NewSubScope(name);
    auto upsampling = ResizeNearestNeighbor(scope.WithOpName("ResizeArea-" + name), input,
                                            Const(t_root.WithOpName("const_resize_area" + name),
                                                  {imageWidth / upsamplingCoenf, imageWidth / upsamplingCoenf}));
    auto conv = AddConv2D(scope, upsampling, k, inputN, outputN, s, name, true);
    auto relu = Relu(scope.WithOpName("Relu-" + name), conv);
    return Add(scope.WithOpName("Add-" + name), relu, down);
}

Status InvertibleGrayscale::CreateEncodeGraph() {
    //input image is batch_sizex150x150x3
    input_batch_var = Placeholder(t_root.WithOpName("input_encode"), DT_FLOAT,
                                  Placeholder::Attrs().Shape({8, imageWidth, imageWidth, 3}));

    auto l = firstLayer(input_batch_var);
    auto l1 = residualLayer(l, 3, 64, 64, 1, "e1", false);
    auto l2 = residualLayer(l1, 3, 64, 64, 1, "e2", false);
    auto l3 = downLayer(l2, 3, 64, 128, 1, "e3");
    auto l4 = downLayer(l3, 3, 128, 256, 1, "e4");
    auto l5 = residualLayer(l4, 3, 256, 256, 1, "e5", false);
    auto l6 = residualLayer(l5, 3, 256, 256, 1, "e6", false);
    auto l7 = residualLayer(l6, 3, 256, 256, 1, "e7", false);
    auto l8 = residualLayer(l7, 3, 256, 256, 1, "e8", false);
    auto l9 = upLayer(l8, l3, 3, 256, 128, 1, 2, "e9");
    auto l10 = upLayer(l9, l2, 3, 128, 64, 1, 1, "e10");
    auto l11 = residualLayer(l10, 3, 64, 64, 1, "e11", false);
    auto l12 = residualLayer(l11, 3, 64, 64, 1, "e12", false);
    auto l13 = lastLayer(l12, 1, "e13");
    auto l14 = Abs(t_root.WithOpName("e14-abs"), l13);
    grayImage = l14;
    //Multiply(t_root.WithOpName("e14-multi"), l14, {255.0f});
//    newColorImage = grayImage;

//    return t_root.status();

    return CreateDecodeGraph(l14);
}

Status InvertibleGrayscale::CreateDecodeGraph(Input input) {
    //input image is batch_sizex150x150x3
    //Start Conv+Maxpool No 1. filter size 3x3x3 and we have 32 filters
    Scope scope = t_root.NewSubScope("decode-resize");
    auto conv = AddConv2D(scope, input, 3, 1, 64, 1, "d0", false);
    auto l1 = residualLayer(conv, 3, 64, 64, 1, "d1", true);
    auto l2 = residualLayer(l1, 3, 64, 64, 1, "d2", true);
    auto l3 = residualLayer(l2, 3, 64, 64, 1, "d3", true);
    auto l4 = residualLayer(l3, 3, 64, 64, 1, "d4", true);
    auto l5 = residualLayer(l4, 3, 64, 64, 1, "d5", true);
    auto l6 = residualLayer(l5, 3, 64, 64, 1, "d6", true);
    auto l7 = residualLayer(l6, 3, 64, 64, 1, "d7", true);
    auto l8 = residualLayer(l7, 3, 64, 64, 1, "d8", true);
    auto l9 = lastLayer(l8, 3, "d9");
    auto l10 = Abs(t_root.WithOpName("d10-abs"), l9);
    newColorImage = l9;//Multiply(t_root.WithOpName("d10-multi"), l10, {255.0f});
    return t_root.status();
}


Status InvertibleGrayscale::Initialize() {
    if (!t_root.ok())
        return t_root.status();

    vector<Output> ops_to_run;
    for (pair<string, Output> i: m_assigns)
        ops_to_run.push_back(i.second);
    t_session = unique_ptr<ClientSession>(new ClientSession(t_root));
    TF_CHECK_OK(t_session->Run(ops_to_run, nullptr));
    return Status::OK();
}

Status InvertibleGrayscale::TrainCNN(Tensor &image_batch, vector<Tensor> &results, int mode) {
    if (!t_root.ok())
        return t_root.status();

    //Inputs: batch of images, labels, drop rate and do not skip drop.
    //Extract: Loss and result. Run also: Apply Adam commands

    auto grayBatch = getLuminace(image_batch);
    if (mode == 1) {
        TF_CHECK_OK(t_session->Run(
                {{input_batch_var, image_batch},
                 {input_gray_var,  grayBatch}
                },
                {grayImage, newColorImage, loss_op1},
                out_grads_op1,
                &results));
    } else if (mode == 2) {
        TF_CHECK_OK(t_session->Run(
                {{input_batch_var, image_batch},
                 {input_gray_var,  grayBatch}},
                {grayImage, newColorImage, loss_op1},
                out_grads_op2,
                &results));
    }

    return Status::OK();
}

Status InvertibleGrayscale::Predict(Tensor &image, vector<Tensor> &result, int direction) {
    if (!t_root.ok())
        return t_root.status();
    if (isLoaded) {
        if (direction == 0) {
            tensor_dict feed_dict2 = {
                    {"input/batch", image}
            };
            TF_CHECK_OK(LoadedBundle.GetSession()->Run(feed_dict2, {"encode/latent/Tanh"}, {}, &result));
            return tensorflow::Status::OK();
        } else {
            tensor_dict feed_dict2 = {
                    {"encode/latent/Tanh", image}
            };
            TF_CHECK_OK(LoadedBundle.GetSession()->Run(feed_dict2, {"decode/Tanh"}, {}, &result));
            return tensorflow::Status::OK();
        }
    } else {
        if (direction == 0) {
            TF_CHECK_OK(t_session->Run({{input_batch_var, image}}, {grayImage}, &result));
        }else{
            TF_CHECK_OK(t_session->Run({{grayImage, image}}, {newColorImage}, &result));

        }
    }
    return Status::OK();
}

Status InvertibleGrayscale::Loss(Scope scope) {
    /// contrast loss
    ///    target_224 = tf.image.resize_images(target_imgs, size=[224, 224], method=0, align_corners=False)
    ///    predict_224 = tf.image.resize_images(latent_imgs, size=[224, 224], method=0, align_corners=False)
    ///    vgg19_api = VGG19("../vgg19.npy")
    ///    vgg_map_targets = vgg19_api.build((target_224 + 1) / 2, is_rgb=True)
    ///    vgg_map_predict = vgg19_api.build((predict_224 + 1) / 2, is_rgb=False)
    ///# stretch the global contrast to follow color contrast
    ///    vgg_loss = 1e-7 * tf.losses.mean_squared_error(vgg_map_targets, vgg_map_predict)
    ///local structer loss
    ///    LoadLibrary("./CustomLayer/zero_out.so", nullptr, nullptr, nullptr);
//    auto target_224 = ResizeBilinear(scope.WithOpName("vgg_resize_target"), input_batch_var, {224, 224});
//    auto predict_224 = ResizeBilinear(scope.WithOpName("vgg_resize_predict"), grayImage, {224, 224});
//    auto vgg19 = new VGG19(scope);
//    auto vgg_target = vgg19->getVgg19(scope, target_224, true);
//    auto vgg_predict = vgg19->getVgg19(scope, predict_224, false);
//
//    auto vgg_loss = MeanAxis(scope, SquaredDifference(scope, vgg_target, vgg_predict), {1, 0}, "vgg_mse");
    ///    # quantization loss
    ///    latent_stack = tf.concat([latent_imgs for t in range(256)], axis=3)
    ///    id_mat = np.ones(shape=(1, 1, 1, 1))
    ///    quant_stack = np.concatenate([id_mat * t for t in range(256)], axis=3)
    ///    quant_stack = (quant_stack / 127.5) - 1
    ///    quantization_map = tf.reduce_min(tf.abs(latent_stack - quant_stack), axis=3)
    ///    quantization_loss = tf.reduce_mean(quantization_map)
//
//    auto quantStack = (Input::Initializer({1}, {1, 1, 1, 256}));
//    auto quantStack_tensor =  quantStack.tensor.tensor<float, 4>();
//    for (int i = 0; i < 256; i++) {
//        quantStack_tensor(0,0,0,i)=float(i);
//    }
//    auto ql_quantSub = Subtract(scope.WithOpName("ql_sbu"), Div(scope.WithOpName("ql_sub_div"), quantStack, 127.5f),
//                                1.0f);
//    vector<Output> grayImageStack(256, grayImage);
//    auto ql_latentStack = Concat(scope.WithOpName("ql_concat"), grayImageStack, 3);
//    auto ql_abs = Abs(scope.WithOpName("ql_abs"),
//                      Subtract(scope.WithOpName("ql_abs_sub"), ql_latentStack, ql_quantSub));
//    auto ql_map = Min(scope.WithOpName("ql_min"), ql_abs, {3});
//    auto quantization_loss = Mean(scope.WithOpName("ql_mean"), ql_map, {0});
    /// local structer loss
    auto tvp = TotalVariation(scope.WithOpName("grads_loss_tv_pred"), grayImage, "predicated_gl");
    auto tvt = TotalVariation(scope.WithOpName("grads_loss_tv_target"), input_gray_var, "target_gl");

    auto latent_grads = MeanAxis(
            scope,
            Div(scope.WithOpName("gl_mean_pred_div"), tvp, float(imageWidth * imageWidth)),
            {1, 0},
            "grads_loss_mean_pred"
    );
    auto target_grads = MeanAxis(
            scope,
            Div(scope.WithOpName("gl_mean_pred_div"), tvt, float(imageWidth * imageWidth)),
            {1, 0},
            "grads_loss_mean_target"
    );

    auto grads_loss = Abs(scope.WithOpName("grads_loss_abs"),
                          Subtract(scope.WithOpName("grads_loss_sub"), latent_grads, target_grads));
    ///lightness Loss
    /// control the mapping order similar to normal rgb2gray
    ///    global_order_loss = tf.reduce_mean(tf.maximum(70 / 127.0, tf.abs(gray_inputs - latent_imgs))) - 70 / 127.0
    auto gol_abs = Abs(scope.WithOpName("gol_abs"),
                       Subtract(scope.WithOpName("gol_abs_sub"), input_gray_var, grayImage));
    auto gol_max = Maximum(scope.WithOpName("gol_maximum"), 70 / 127.0f, gol_abs);
    auto gol_mean = MeanAxis(scope, gol_max, {3, 2, 1, 0}, "gol_mean");
    auto global_order_loss = Subtract(scope.WithOpName("gol_substract"), gol_mean, 70 / 127.0f);
    ///    mse_loss = tf.losses.mean_squared_error(target_imgs, pred_imgs)
    //Invertibility loss
    auto mse_loss = MeanAxis(scope, SquaredDifference(scope, newColorImage, input_batch_var), {3, 2, 1, 0}, "mse");
    ///    loss_op1 = 3 * mse_loss + vgg_loss + 0.5 * grads_loss + global_order_loss
    ///    loss_op2 = 3 * mse_loss + vgg_loss + 0.1 * grads_loss + global_order_loss + 10 * quantization_loss
    auto mse_lossWC = Multiply(scope.WithOpName("loss_mlp"), mse_loss, 3.0f);
//    auto mseVgg = Add(scope.WithOpName("loss_add"), mse_lossWC, vgg_loss);
    auto mseVggGlobal = Add(scope.WithOpName("loss_add"), mse_lossWC, global_order_loss);

    auto gradsOp1 = Multiply(scope.WithOpName("loss_mlp_op1_g"), grads_loss, 0.5f);
    auto gradsOp2 = Multiply(scope.WithOpName("loss_mlp_op2_g"), grads_loss, 0.1f);
    loss_op1 = Add(scope.WithOpName("loss_add"), mseVggGlobal, gradsOp1);
//    auto quantWC = Multiply(scope.WithOpName("loss_mlp"), quantization_loss, 10.0f);
//    auto mseVggGlobalQuant = Add(scope.WithOpName("loss_add"), mseVggGlobal, quantWCl);
    loss_op2 = Add(scope.WithOpName("loss_add"), mseVggGlobal, gradsOp2);
    return scope.status();

}

Output InvertibleGrayscale::TotalVariation(Scope scope, Input input, string name) {
//    Conv2D(scope.WithOpName("Conv" + name), input, m_vars["W" + name], {1, s, s, 1}, "SAME")
    auto x1 = ScaleAndTranslate(scope.WithOpName("scale-x1-" + name), input,
                                {imageWidth, imageWidth}, {1.0f, 1.0f}, {0.0f, 0.0f,});
    auto y1 = ScaleAndTranslate(scope.WithOpName("scale-y1-" + name), input,
                                {imageWidth, imageWidth}, {1.0f, 1.0f}, {0.0f, 0.0f});
    auto absSubX1 = Abs(scope.WithOpName("abs-x1-" + name),
                        Subtract(scope.WithOpName("subtrack-x1-" + name), input, x1));
    auto absSubY1 = Abs(scope.WithOpName("abs-y1-" + name),
                        Subtract(scope.WithOpName("subtrack-y1-" + name), input, y1));
    auto add = Add(scope.WithOpName("add-" + name), absSubX1, absSubY1);
    auto kd = Sum::Attrs();
    auto sum = Sum(scope.WithOpName("sum-" + name + "-2"), add, {3, 2}, kd.KeepDims(false));
    //    auto out2 = Sum(scope.WithOpName("sum-" + name + "-2"), sum, {3}, kd.KeepDims(false));
    sum = Sum(scope.WithOpName("sum-" + name + "-1"), sum, {0, 1}, kd.KeepDims(true));
    return sum;
}

Output InvertibleGrayscale::MeanAxis(Scope scope, Input input, vector<int> axis, string name) {
    auto kd = Mean::Attrs();
    if (axis.size() == 1) {
        return Mean(scope.WithOpName(name + to_string(axis[0])), input, {axis[0]}, kd.KeepDims(false));
    } else if (axis.size() > 1) {
        auto out = Mean(scope.WithOpName(name + to_string(axis[0])), input, {axis[0]}, kd.KeepDims(false));
        for (int i = 1; i < axis.size() - 1; i++) {
            out = Mean(scope.WithOpName(name + to_string(axis[i])), out, {axis[i]}, kd.KeepDims(false));
        }
        out = Mean(scope.WithOpName(name + to_string(axis[axis.size() - 1])), out, {axis[axis.size() - 1]},
                   kd.KeepDims(false));
        return out;
    }
}

Tensor InvertibleGrayscale::getLuminace(Tensor &trueColorImage) {
    const TensorShape &input_shape = trueColorImage.shape(); //get the shape of the tensor
    auto input_tensor = trueColorImage.tensor<float, 4>();//true conversion
    Tensor output(trueColorImage.dtype(),
                  {input_shape.dim_size(0), input_shape.dim_size(1), input_shape.dim_size(2), 1});
    auto output_tensor = output.tensor<float, 4>();//true conversion
    float luminance = 0;
    for (int batch = 0; batch < input_shape.dim_size(0); batch++) {
        for (int i = 0; i < input_shape.dim_size(1); i++) {
            for (int j = 0; j < input_shape.dim_size(2); j++) {
                luminance = 0.2989 * input_tensor(batch, i, j, 0) + 0.5870 * input_tensor(batch, i, j, 1) +
                            0.1140 * input_tensor(batch, i, j, 2);
                output_tensor(batch, i, j, 0) = luminance;
            }
        }
    }
    return output;
}


Status InvertibleGrayscale::LoadGraphFromFile(string path) {
    Status load_status = LoadSavedModel(
            SessionOptions(), RunOptions(), path, {"serve"}, &LoadedBundle);
    isLoaded = true;
    ofstream myfile;
    myfile.open("output.cpp");
    myfile << LoadedBundle.meta_graph_def.DebugString();
    myfile.close();
    return load_status;

}

Status InvertibleGrayscale::BuildAndTrain(string base_folder, int batch_size) {
    auto imageLoader = ImageLoader(imageWidth, imageHeight, imageChannels);
    Status s = imageLoader.CreateGraphForImage(true);
    vector<Tensor> image_batches;
    s = imageLoader.ReadBatches(base_folder, batch_size, image_batches);
    TF_CHECK_OK(s);
    //CNN model
    isLoaded = false;
    s = this->CreateEncodeGraph();
    TF_CHECK_OK(s);
    size_t num_batches = image_batches.size();
    float epsilon = 1.0f / num_batches * batch_size * 120.0f;
    s = this->CreateOptimizationGraph(0.0002f, epsilon);//input is learning rate
    TF_CHECK_OK(s);
    //Run inititialization
    s = this->Initialize();

    TF_CHECK_OK(s);
    int num_epochs = 2;
    //Epoch / Step loops
    wprintf(L"Start Training: %d\n",num_batches);

    for (int epoch = 0; epoch < 90; epoch++) {
        wprintf(L"Epoch %d /90\n", epoch);
        for (int b = 0; b < 1; b++) {
            wprintf(L"Start image no.: %d\n" ,b );
            vector<Tensor> results;
            auto grayBatch = this->getLuminace(image_batches[b]);
            s = this->TrainCNN(image_batches[b], results, 1);
        }
    }
    for (int epoch = 0; epoch < 30; epoch++) {
        wprintf(L"Epoch %d /30\n",epoch);
        for (int b = 0; b < num_batches; b++) {
            wprintf(L"Start image no.: %d \n" ,b );
            vector<Tensor> results;
            auto grayBatch = this->getLuminace(image_batches[b]);
            s = this->TrainCNN(image_batches[b], results, 2);
        }
    }
//    //testing the model
//    s = imageLoader.CreateGraphForImage(false);//rebuild the model without unstacking
//    TF_CHECK_OK(s);
//    base_folder = "../TMOMeng18/data/test";
//    vector<pair<Tensor, string>> all_files_tensors;
//    s = imageLoader.ReadFileTensors(base_folder, all_files_tensors);
//    TF_CHECK_OK(s);
//    //test a few images
//    int count_images = 10;
//    int count_success = 0;
//    for (int i = 0; i < count_images; i++) {
//        pair<Tensor, string> p = all_files_tensors[i];
//        vector<Tensor> result;
//        s = this->Predict(p.first, result,0);
//    }
    wprintf(L"successes\n");
    return t_root.status();
}

Status InvertibleGrayscale::Build() {
    isLoaded = false;
    Status s = this->CreateEncodeGraph();
    //Run inititialization
    s = this->Initialize();
    TF_CHECK_OK(s);
    return t_root.status();
}

Status InvertibleGrayscale::SaveTrainedGraph(string path) {
    //save
//    vector<Tensor> out_tensors;
//    //Extract: current weights and biases current values
////    TF_CHECK_OK(t_session->Run(v_weights_biases , &out_tensors));
//    unordered_map<string, Tensor> variable_to_value_map;
//    int idx = 0;
//    for(Output o:  m_vars)
//    {
//        variable_to_value_map[o.node()->name()] = out_tensors[idx];
//        idx++;
//    }
//    GraphDef graph_def;
//    TF_CHECK_OK(t_root.ToGraphDef(&graph_def));
//    //call the utility function (modified)
//    SavedModelBundle saved_model_bundle;
////    SignatureDef signature_def;
////    (*signature_def.mutable_inputs())[input_batch_var.name()].set_name(input_batch_var.name());
////    (*signature_def.mutable_outputs())[out_classification.name()].set_name(out_classification.name());
////    MetaGraphDef* meta_graph_def = &saved_model_bundle.meta_graph_def;
////    (*meta_graph_def->mutable_signature_def())["signature_def"] = signature_def;
////    *meta_graph_def->mutable_graph_def() = graph_def;
//    SessionOptions session_options;
//    saved_model_bundle.session.reset(NewSession(session_options));//even though we will not use it
    GraphDef g;
    t_root.ToGraphDef(&g);
//    saved_model_bundle.session->Create(g);
//
//    GraphDef frozen_graph_def;
//    std::unordered_set<string> inputs;
//    std::unordered_set<string> outputs;
//    TF_CHECK_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs, &outputs));

    //write to file
    WriteBinaryProto(Env::Default(), "../TMOMeng18/graphs/test/saved_model.pb", g);
    wprintf(L"saved");

//    return t_root.status();
    return this->LoadGraphFromFile("../TMOMeng18/graphs/test");
}

