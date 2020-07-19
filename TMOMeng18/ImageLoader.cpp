#include "ImageLoader.h"

Status ImageLoader::CreateGraphForImage(bool unstack) {
    file_name_var = Placeholder(i_root.WithOpName("input"), DT_STRING);
    auto file_reader = ReadFile(i_root.WithOpName("file_readr"), file_name_var);

    auto image_reader = DecodeJpeg(i_root.WithOpName("jpeg_reader"), file_reader,
                                   DecodeJpeg::Channels(image_channels));

    auto float_caster = Cast(i_root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    auto dims_expander = ExpandDims(i_root.WithOpName("dim"), float_caster, 0);
    auto resized = ResizeBilinear(i_root.WithOpName("size"), dims_expander,
                                  Const(i_root, {imageWidth, imageHeight}));
    auto div = Div(i_root.WithOpName("normalized"), resized, {255.f});

    if (unstack) {
        auto output_list = Unstack(i_root.WithOpName("fold"), div, 1);
        image_tensor_var = output_list.output[0];
    } else{
        image_tensor_var = div;
    }
    return i_root.status();
}

Status ImageLoader::ReadTensorFromImageFile(string &file_name, Tensor &outTensor) {
    if (!i_root.ok())
        return i_root.status();
    if (!str_util::EndsWith(file_name, ".jpg") && !str_util::EndsWith(file_name, ".jpeg")) {
        return errors::InvalidArgument("Image must be jpeg encoded");
    }
    vector<Tensor> out_tensors;
    ClientSession session(i_root);
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {image_tensor_var}, &out_tensors));
    outTensor = out_tensors[0]; // shallow copy
    return Status::OK();
}

Status ImageLoader::ReadFileTensors(string &folder_name, vector<pair<Tensor, string>> &file_tensors) {
    //validate the folder
    Env *penv = Env::Default();
    //get the files
    bool b_shuffle = false;
    TF_RETURN_IF_ERROR(penv->IsDirectory(folder_name));
    vector<string> file_names;
    TF_RETURN_IF_ERROR(penv->GetChildren(folder_name, &file_names));
    for (string file: file_names) {
        string full_path = io::JoinPath(folder_name, file);
        Tensor i_tensor;
        wprintf(L"%s\n",full_path.c_str());
        TF_RETURN_IF_ERROR(ReadTensorFromImageFile(full_path, i_tensor));
        file_tensors.emplace_back(i_tensor, full_path);
    }
    b_shuffle = true;
    return Status::OK();
}

Status ImageLoader::ReadBatches(string folder_name, int batch_size,
                                vector<Tensor> &image_batches) {
    vector<pair<Tensor, string>> all_files_tensors;
    TF_RETURN_IF_ERROR(ReadFileTensors(folder_name, all_files_tensors));
    auto start_i = all_files_tensors.begin();
    auto end_i = all_files_tensors.begin() + batch_size;
    size_t batches = all_files_tensors.size() / batch_size;
    if (batches * batch_size < all_files_tensors.size())
        batches++;
    for (int b = 0; b < batches; b++) {
        if (end_i > all_files_tensors.end())
            end_i = all_files_tensors.end();
        vector<pair<Tensor, string>> one_batch(start_i, end_i);
        //need to break the pairs
        vector<Input> one_batch_image;
        for (auto p: one_batch) {
            one_batch_image.push_back(Input(p.first));
        }
        InputList one_batch_inputs(one_batch_image);
        Scope root = Scope::NewRootScope();
        auto stacked_images = Stack(root, one_batch_inputs);
        ClientSession session(root);
        TF_CHECK_OK(root.status());
        vector<Tensor> out_tensors;
        TF_CHECK_OK(session.Run({}, {stacked_images, stacked_images}, &out_tensors));
        image_batches.push_back(out_tensors[0]);
        start_i = end_i;
        if (start_i == all_files_tensors.end())
            break;
        end_i = start_i + batch_size;
    }
    return Status::OK();
}