# import the necessary packages
from helper import FACIAL_LANDMARKS_68_IDXS
from helper import FACIAL_LANDMARKS_5_IDXS
from helper import shape_to_np, rect_to_bb
import numpy as np
import cv2
from skimage import transform as tf

def crop_image(image, detector, predictor):
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)

        shape = ((shape - np.array([new_x,new_y])) * scale)

        return roi, shape 

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect, shape, scale=None):
        # convert the landmark (x, y)-coordinates to a NumPy array
        # shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        
        #simple hack ;)
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
            
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        if scale is None:
            scale = 1.2 * desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output, scale

    def get_tform(self, image, shape, mean_shape, scale=None):
        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]

        leftEyeCenter = mean_shape[left_eye, :].mean(axis=0)
        rightEyeCenter = mean_shape[right_eye, :].mean(axis=0)
        noseCenter = mean_shape[nose, :].mean(axis=0)

        # print(leftEyeCenter)
        # exit()

        template_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])

        leftEyeCenter = shape[left_eye, :].mean(axis=0)
        rightEyeCenter = shape[right_eye, :].mean(axis=0)
        noseCenter = shape[nose, :].mean(axis=0)

        dst_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        tform = tf.SimilarityTransform()
        tform.estimate( template_points, dst_points)

        self.tform = tform

    def apply_tform(self, image):
        output = tf.warp(image, self.tform, output_shape=(self.desiredFaceWidth, self.desiredFaceHeight))
        output = (output*255).astype('uint8')

        return output, None

    def align_three_points(self, image, shape, mean_shape, scale=None):
        # shape = shape_to_np(shape)

        # (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        # (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        # (nStart, nEnd) = FACIAL_LANDMARKS_68_IDXS["nose"]

        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]

        leftEyeCenter = mean_shape[left_eye, :].mean(axis=0)
        rightEyeCenter = mean_shape[right_eye, :].mean(axis=0)
        noseCenter = mean_shape[nose, :].mean(axis=0)

        # print(leftEyeCenter)
        # exit()

        template_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])

        leftEyeCenter = shape[left_eye, :].mean(axis=0)
        rightEyeCenter = shape[right_eye, :].mean(axis=0)
        noseCenter = shape[nose, :].mean(axis=0)

        dst_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        tform = tf.SimilarityTransform()
        tform.estimate( template_points, dst_points)

        output = tf.warp(image, tform, output_shape=(self.desiredFaceWidth, self.desiredFaceHeight))
        output = (output*255).astype('uint8')

        return output, None

    def align_box(self, shape, scale=None):
        
        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]
        all_pts = nose + left_eye + right_eye

        mean_pts = shape[all_pts, :].mean(axis=0).astype(int)
        
        return mean_pts

