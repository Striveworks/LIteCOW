import grpc
from litecow.client import ICOWClient

import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image


class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.float32(boxed_image)
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def postprocess(indices, scores, boxes):
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])

    return out_boxes, out_scores, out_classes




# get all of the results above a threshold
def get_boxes(out_boxes, out_scores, out_classes):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for idx, _ in enumerate(out_boxes):
        box = BoundBox(out_boxes[idx][1], out_boxes[idx][0], out_boxes[idx][3], out_boxes[idx][2])
        v_boxes.append(box)
        v_labels.append(labels[out_classes[idx]])
        v_scores.append(out_scores[idx]*100)
    return v_boxes, v_labels, v_scores



# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	data = pyplot.imread(filename)
	pyplot.imshow(data)
	ax = pyplot.gca()
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		width, height = x2 - x1, y2 - y1
		rect = Rectangle((x1, y1), width, height, fill=False, color='blue')
		ax.add_patch(rect)
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='white')
	pyplot.show()


def main():
    img_path = './cow.jpeg'

    image = Image.open(img_path)
    # input
    image_data = preprocess(image)
    image_size = np.array([image.size[1], image.size[0]]).reshape(1, 2)

    channel = grpc.insecure_channel("icow-service.icow.127.0.0.1.nip.io:80")
    client = ICOWClient(channel)

    feed_f = dict(zip(['input_1', 'image_shape'],(image_data, np.float32(np.array([image.size[1], image.size[0]]).reshape(1, 2)))))
    results = client.get_inference("s3://models/tinyyolov3", feed_f)

    boxes, scores, indices = results['yolonms_layer_1'], results['yolonms_layer_1:1'], results['yolonms_layer_1:2']
    out_boxes, out_scores, out_classes = postprocess(indices, scores, boxes)
    v_boxes, v_labels, v_scores = get_boxes(out_boxes, out_scores, out_classes)

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(img_path, v_boxes, v_labels, v_scores)

    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])


if __name__ == '__main__':
     main()
