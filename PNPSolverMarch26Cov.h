#ifndef BACKEND_PNPSOLVER_H_
#define BACKEND_PNPSOLVER_H_

#include <isam/Pose3d.h>
#include "KinectCamera.h"
#include "Surf3DTools.h"

class PNPSolver
{
    public:
        PNPSolver(KinectCamera * KinectCamera);
        virtual ~PNPSolver();

        void getRelativePose(isam::Pose3d &pose,
                             std::vector<std::pair<int2, int2> > & inliers,
                             std::vector<InterestPoint *> & scene,
                             std::vector<InterestPoint *> & model,
                             Eigen::MatrixXd & cov);

    private:
        KinectCamera * camera;
};

#endif /* BACKEND_PNPSOLVER_H_ */
