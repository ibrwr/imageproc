import importlib
from distutils.version import LooseVersion
import SimpleITK as sitk
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, interactive
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from myshow import myshow, myshow3d
%matplotlib inline


def __return_image(input_object):
  if isinstance(input_object, sitk.Image):
    return input_object
  else:
    return input_object.output_image

def __return_mask(input_object):
  image = __return_image(input_object)
  cast = sitk.Cast(image, sitk.sitkUInt8)
  return sitk.BinaryThreshold(cast, lowerThreshold=1)


def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("BMPImageIO")
    reader.SetFileName(path)
    rgb_image = reader.Execute()
    image = sitk.VectorIndexSelectionCast(rgb_image,0)
    return image

def show_overlay(scan, overlay):
  overlay_image = __return_image(overlay)
  myshow(sitk.LabelOverlay(scan, overlay_image), "Result")

def plot_hist(image):
    plt.hist(image, bins=100)
    plt.title("Histogram intensity values")
    plt.show()
    return

def calculate_dice(reference_segmentation, my_segmentation):
  ref_seg = __return_mask(reference_segmentation)
  seg = __return_mask(my_segmentation)
  myshow(ref_seg)
  myshow(seg)
  overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
  overlap_measures_filter.Execute( ref_seg, seg)
  dice = overlap_measures_filter.GetDiceCoefficient()
  print(f'Dice overlap = {dice}')
  return


def fill_holes_image(object_to_fill):
  image = __return_image(object_to_fill)
  filter = sitk.BinaryFillholeImageFilter()
  filled_image = filter.Execute(image)
  return filled_image


class ImageProcessing:
  def __init__(self, previous_processing_step):
    if isinstance(previous_processing_step, sitk.Image):
      # There is no previous processing step, only
      # an input image
      self.input_image = previous_processing_step
      self.prev_proc = None
      self.output_image = 0 * self.input_image
    else:
      self.prev_proc = previous_processing_step
      self.output_image = 0 * self.prev_proc.output_image
    self.interactive()

  def interactive(self):
      raise NotImplementedError("Must override method interactive")

  def _input_image(self):
    if self.prev_proc:
      return self.prev_proc.output_image
    else:
      return self.input_image



class Thresholding(ImageProcessing):
  def __init__(self, previous_processing_step, lower_threshold=50, upper_threshold=250):
    self.lower_threshold = lower_threshold
    self.upper_threshold = upper_threshold
    super().__init__(previous_processing_step)


  def threshold_image(self, min_thres, max_thresh):
    self.output_image = sitk.BinaryThreshold(self._input_image(), lowerThreshold=min_thres, upperThreshold=max_thresh)

  def interactive(self):

      # Create an interactive slider for thresholding
      threshold_slider_max = widgets.IntSlider(value=self.upper_threshold, min=1, max=max(self._input_image()), step=1, description='Upper Threshold')
      threshold_slider_min = widgets.IntSlider(value=self.lower_threshold, min=1, max=max(self._input_image()), step=1, description='Lower Threshold')

      # Use the interactive function to update the threshold based on the slider value
      widget = interact(self._update, threshold_value_max=threshold_slider_max, threshold_value_min=threshold_slider_min)

      return widget

  def _update(self, threshold_value_max, threshold_value_min):
          # Prevent lower bound to be higher than upperbound (error prevention).
          if threshold_value_min > threshold_value_max:
              threshold_value_min = threshold_value_max

          # Threshold the image
          self.threshold_image(threshold_value_min, threshold_value_max)

          # Display the thresholded image
          myshow(self.output_image, title=f'Threshold range: {threshold_value_min} - {threshold_value_max}')
          #return result_image


class MorphologicalProcessing(ImageProcessing):
  def __init__(self, previous_processing_step, kernel_size=0):
    self.kernel_size = kernel_size
    super().__init__(previous_processing_step)

  def interactive(self):
    kernel_size_slider = widgets.IntSlider(value=self.kernel_size, min=0, max=5, step=1, description='Kernel Size')
    interact(self._update, kernel_size=kernel_size_slider)

  def _update(self, kernel_size):
      self.process_image(kernel_size)
      self.kernel_size = kernel_size
      myshow(self.output_image, title=self.plot_title)

  def process_image(self):
      raise NotImplementedError("Must override method process_image")


class Erosion(MorphologicalProcessing):
  plot_title = 'Erosion'

  def process_image(self, kernel_size):
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelType(sitk.sitkBall)
    erode_filter.SetKernelRadius(kernel_size)
    self.output_image = erode_filter.Execute(self._input_image())


class Dilation(MorphologicalProcessing):
  plot_title = 'Dilation'

  def process_image(self, kernel_size):
      dilate_filter = sitk.BinaryDilateImageFilter()
      dilate_filter.SetKernelType(sitk.sitkBall)
      dilate_filter.SetKernelRadius(kernel_size)
      self.output_image = dilate_filter.Execute(self._input_image())


class Closing(MorphologicalProcessing):
  plot_title = 'MorphologicalClosing'

  def process_image(self, kernel_size):
    filter = sitk.BinaryMorphologicalClosingImageFilter()
    filter.SetKernelRadius([kernel_size, kernel_size])
    self.output_image = filter.Execute(self._input_image())


class ConnectedComponents(ImageProcessing):

  #def __init__(self, ImageProcessing):
    #self.large_components = 0 * sitk.Image(input_image)
   # ImageProcessing.__init__(self, input_image)

  def find_large_components(self, min_size = [10,10]):
    conn_comp = sitk.ConnectedComponent(self._input_image())
    filtered_comp = sitk.Image(conn_comp)
    filtered_comp[:,:] = 0

    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(conn_comp)

    for comp in range(1, max(conn_comp)):
      comp_size = lsif.GetBoundingBox(comp)[2:4]
      larger_than_min = all(c > m for c,m in zip(comp_size,min_size))
      if larger_than_min:
        filtered_comp[conn_comp==comp] = 1

    #myshow(filtered_comp, "All bigger connected components")
    self.large_components = sitk.ConnectedComponent(filtered_comp)

  def choose_connected_component(self, component_id):
      # Create an equally sized image with everything blank except the wanted component.
      self.output_image = sitk.Image(self.large_components)
      self.output_image[:,:] = 0
      self.output_image[self.large_components==component_id] = 1


  def interactive(self):

      self.find_large_components()

      # Create an interactive slider for thresholding
      component_slider = widgets.IntSlider(value=1, min=0, max=max(10,max(self.large_components)), step=1, description='Component')

      # Use the interactive function to update the threshold based on the slider value
      widget = interact(self._update, component_id=component_slider)

      return widget

  def _update(self, component_id):
      self.find_large_components()
      self.choose_connected_component(component_id)
      myshow(self.output_image, title=f'Component: {component_id}')


