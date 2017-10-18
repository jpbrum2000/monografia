#ifndef _BIC_H_
#define _BIC_H_

#include "cimage.h"
#include "histogram.h"
#include "featurevector.h"

//Histogram *BIC(CImage *cimg, Image *mask);
FeatureVector1DInt *BIC(CImage *cimg, Image *mask);

#endif
