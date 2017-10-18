#include "libcolordescriptors.h"

int main(int argc, char** argv)
{
  FeatureVector1DInt *bic1=NULL;
  FeatureVector1DInt *bic2=NULL;
  double distance;

  if (argc != 3) {
    fprintf(stderr,"usage: bic_distance fv1_path fv2_path\n");
    exit(-1);
  }

  bic1 = ReadFileFeatureVector1DInt(argv[1]);
  bic2 = ReadFileFeatureVector1DInt(argv[2]);

  distance = (double) EuclideanDistanceInt(bic1, bic2);
  printf("%lf\n",distance);

  DestroyFeatureVector1DInt(&bic1);
  DestroyFeatureVector1DInt(&bic2);

  return(0);
}
