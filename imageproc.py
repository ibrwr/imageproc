import importlib
from distutils.version import LooseVersion
import SimpleITK as sitk
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, interactive
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ITK.Python.myshow import myshow, myshow3d
import io
import base64


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


def init_output():
    plt.ioff()
    ImageProcessing.html_widgets = []


def show_output():
    display(widgets.VBox(ImageProcessing.html_widgets))

# Output is organized in html widget to prevent an update of a slider
# see https://stackoverflow.com/questions/31911884/interactive-plots-placement-in-ipython-notebook-widget
class ImageProcessing:

  html_widgets = []

  def __init__(self, previous_processing_step):
    if isinstance(previous_processing_step, sitk.Image):
      # There is no previous processing step, only
      # an input image
      self.input_image = previous_processing_step
      self.prev_proc = None
      self.output_image = 0 * self.input_image
    else:
      previous_processing_step.next_proc = self
      self.prev_proc = previous_processing_step
      self.output_image = 0 * self.prev_proc.output_image
    self.next_proc = None
    self.html_for_plot = widgets.HTML()
    #self.fig=None
    #self.ax=None
    self.interactive()
    self._add_widget(self.html_for_plot)
    self._refresh()


  def interactive(self):
      raise NotImplementedError("Must override method interactive")

  def _add_widget(self, widget):
    #print(f'Adding to html_widgets {len(ImageProcessing.html_widgets)}')
    ImageProcessing.html_widgets.append(widget)
    return widget

  def _plot_to_html(self):
    # write image data to a string buffer and get the PNG image bytes
    buf = io.BytesIO()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    #plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    self.html_for_plot.value = """<img src='data:image/png;base64,{}'/>""".format(base64.b64encode(buf.getvalue()).decode('ascii'))
    plt.close()

  def _input_image(self):
    if self.prev_proc:
      return self.prev_proc.output_image
    else:
      return self.input_image

  def _refresh_next(self):
    if self.next_proc:
      self.next_proc._refresh()

  # basically sitk myplot but saving result to html object
  def plot(self, img, title=None, margin=0.1, dpi=80, cmap="gray"):
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        #if not self.fig:
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        # Make the axis the right size...
        self.ax = self.fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        if z is None:
            self.ax.imshow(nda, extent=extent, interpolation=None, cmap=cmap)
        else:
            self.ax.imshow(nda[z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        self._plot_to_html()
        #plt.show()

    if slicer:
        print('we have a slicer')
        interact(callback, z=(0, nda.shape[0] - 1))
    else:
        callback()



class Thresholding(ImageProcessing):
  def __init__(self, previous_processing_step, lower_threshold=50, upper_threshold=250):
    self.lower_threshold = lower_threshold
    self.upper_threshold = upper_threshold
    super().__init__(previous_processing_step)


  def threshold_image(self):
    self.output_image = sitk.BinaryThreshold(self._input_image(),
                                             lowerThreshold=self.lower_threshold,
                                             upperThreshold=self.upper_threshold)

  def interactive(self):
      # Create an interactive slider for thresholding
      upper_threshold_slider = self._add_widget(widgets.IntSlider(value=self.upper_threshold, min=1, max=max(self._input_image()), step=1, description='Upper Threshold'))
      lower_threshold_slider = self._add_widget(widgets.IntSlider(value=self.lower_threshold, min=1, max=max(self._input_image()), step=1, description='Lower Threshold'))

      # Use the interactive function to update the threshold based on the slider value
      interactive(self._update, lower_threshold=lower_threshold_slider, upper_threshold=upper_threshold_slider)

  def _update(self, lower_threshold, upper_threshold):
          # Prevent lower bound to be higher than upperbound (error prevention).
          if lower_threshold > upper_threshold:
              lower_threshold = upper_threshold
          self.lower_threshold = lower_threshold
          self.upper_threshold = upper_threshold
          self._refresh()


  def _refresh(self):
          # Threshold the image
          self.threshold_image()

          # Display the thresholded image
          self.plot(self.output_image, title=f'Threshold range: {self.lower_threshold} - {self.upper_threshold}')
          #return result_image
          self._refresh_next()


class MorphologicalProcessing(ImageProcessing):
  def __init__(self, previous_processing_step, kernel_size=0):
    self.kernel_size = kernel_size
    super().__init__(previous_processing_step)

  def interactive(self):
    kernel_size_slider = self._add_widget(widgets.IntSlider(value=self.kernel_size, min=0, max=5, step=1, description='Kernel Size'))
    interactive(self._update, kernel_size=kernel_size_slider)

  def _update(self, kernel_size):
      self.kernel_size = kernel_size
      self._refresh()


  def _refresh(self):
      self.process_image(self.kernel_size)
      self.plot(self.output_image, title=self.plot_title)
      self._refresh_next()

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

  def __init__(self, previous_processing_step):
    #self.large_components = 0 * sitk.Image(input_image)
    self.component_id = 0
    ImageProcessing.__init__(self, previous_processing_step)

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
      component_slider = self._add_widget(widgets.IntSlider(value=1, min=0, max=max(10,max(self.large_components)), step=1, description='Component'))

      # Use the interactive function to update the threshold based on the slider value
      interactive(self._update, component_id=component_slider)

  def _update(self, component_id):
      self.component_id = component_id
      self._refresh()

  def _refresh(self):
      self.find_large_components()
      self.choose_connected_component(self.component_id)
      self.plot(self.output_image, title=f'Component: {self.component_id}')


