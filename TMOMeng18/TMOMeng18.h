#include "TMO.h"
#include "InvertibleGrayscale.h"
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/framework/ops.h>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
class TMOMeng18 : public TMO  
{
public:
	TMOMeng18();
	virtual ~TMOMeng18();
	virtual int Transform();
protected:
    InvertibleGrayscale* model;
    TMOInt mode;
    TMOString  direction;

    TMOString modelDirPath;
    TMOString dataDirPath;

    Status Predict();
};
