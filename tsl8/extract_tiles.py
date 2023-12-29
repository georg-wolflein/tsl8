from tqdm import tqdm
import os
from os.path import exists
from pathlib import Path
import numpy as np
from PIL import Image
import PIL
from os.path import exists
import glob
from typing import Dict, Tuple, List, Any
import openslide
from concurrent import futures
import time
import multiprocessing as mp
import cv2
from scipy import ndimage
from numpy import ndarray

Image.MAX_IMAGE_PIXELS = None
cores = 16
size = 512
target_mpp = 0.5
svs_path = Path(r"/data/CPTAC-BRCA/")
t_dir = Path(r"/data/CPTAC-BRCA_tiles")
svs_dir = [f for f in svs_path.glob(f'*') if not str(f).endswith(".h5") and not str(f).endswith("logfile")]

def get_patches(img_norm_wsi_jpg, dir_name):
    image_array = np.array(img_norm_wsi_jpg)
    #zoom_levels = [1, 2, 4, 8, 16]

    #for z in zoom_levels:
    z=1
    img_pxl = size * z
    for i in range(0, image_array.shape[0] - img_pxl, img_pxl):
        for j in range(0, image_array.shape[1] - img_pxl, img_pxl):
            patch = image_array[i:i+img_pxl, j:j+img_pxl, :]
            if (np.count_nonzero(patch)/patch.size) >= min(1/2., 4/z**2): #or (z == 1 and np.sum(patch) > 0)
                if z > 1:
                    patch = ndimage.zoom(patch, (size / patch.shape[0], size / patch.shape[1], 1))
                Image.fromarray(patch).save(f"{dir_name}/Tile_{i,j}.jpg")
    
def process_slide_jpg(slide_url):
    slide_name = Path(slide_url)
    slide_cache_dir = slide_name
    if not exists(str(slide_cache_dir)):
        slide_cache_dir.mkdir(parents=True, exist_ok=True)
    
    if (slide_jpg := slide_name/'canny_slide.jpg').exists():
        sl_suff = str(slide_name).split("/")[-1]
        dir_name = sl_suff.split(".")[0]
        if not exists(f"{str(t_dir)}/{dir_name}"):
            os.mkdir(f"{str(t_dir)}/{dir_name}")
            img_norm_wsi_jpg = Image.open(slide_jpg)
            get_patches(img_norm_wsi_jpg, dir_name)

def canny_fcn(patch: np.array) -> Tuple[np.array, bool]:
    patch_img = PIL.Image.fromarray(patch)
    tile_to_greyscale = patch_img.convert('L')
    # tile_to_greyscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as greyscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a greyscale, not colour.

    tile2array = np.array(tile_to_greyscale)

    # hardcoded thresholds
    edge = cv2.Canny(tile2array, 40, 100)

    # avoid dividing by zero
    edge = (edge / np.max(edge) if np.max(edge) != 0 else 0)
    edge = (((np.sum(np.sum(edge)) / (tile2array.shape[0]*tile2array.shape[1])) * 100)
        if (tile2array.shape[0]*tile2array.shape[1]) != 0 else 0)

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    if(edge < 2.):
        #return a black image + rejected=True
        return (np.zeros_like(patch), True)
    else:
        #return the patch + rejected=False
        return (patch, False)


def reject_background(img: np.array, patch_size: Tuple[int,int], step: int, save_tiles: bool = False, outdir: Path = None, cores: int = 8) -> \
Tuple[ndarray, ndarray, List[Any]]:
    img_shape = img.shape
    #print(f"\nSize of WSI: {img_shape}")

    split=True
    x=(img_shape[0]//patch_size[0])*(img_shape[1]//patch_size[1])

    #print(f"Splitting WSI into {x} tiles and Canny background rejection...")
    begin = time.time()
    patches_shapes_list=[]

    with futures.ThreadPoolExecutor(cores) as executor: #os.cpu_count()
        future_coords: Dict[futures.Future, int] = {}
        i_range = range(img_shape[0]//patch_size[0])
        j_range = range(img_shape[1]//patch_size[1])
        for i in i_range:
            for j in j_range:
                patch = img[(i*patch_size[0]):(i*patch_size[0]+step), (j*patch_size[1]):(j*patch_size[1]+step)]
                #(PIL.Image.fromarray(patch)).save(f'{outdir}/patch_{i*len(j_range) + j}.jpg')
                patches_shapes_list.append(patch.shape)
                future = executor.submit(canny_fcn, patch)
                # begin_time_list.append(time.time())
                future_coords[future] = i*len(j_range) + j # index 0 - 3. (0,0) = 0, (0,1) = 1, (1,0) = 2, (1,1) = 3

        del img
        
        #num of patches x 224 x 224 x 3 for RGB patches
        ordered_patch_list = np.zeros((x, patch_size[0], patch_size[1], 3), dtype=np.uint8)
        rejected_tile_list = np.zeros(x, dtype=bool)
        for tile_future in futures.as_completed(future_coords):
            i = future_coords[tile_future]
            #print(f'Received normalised patch #{i} from thread in {time.time()-begin_time_list[i]} seconds')
            patch, is_rejected = tile_future.result()
            ordered_patch_list[i] = patch
            rejected_tile_list[i] = is_rejected


    #print(f"\nFinished Canny background rejection, rejected {np.sum(rejected_tile_list)} tiles: {end-begin}")
    return ordered_patch_list, rejected_tile_list, patches_shapes_list

def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = PIL.Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    zoom_list = []
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(PIL.Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))
                zoom_list.append(1)

    return canny_img, canny_output_array, coords_list, zoom_list

def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224, cores: int = 8) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    steps = 8
    stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        #print(f"Read slide MPP of {slide_mpp} from meta-data")
    except KeyError:
        #if it fails, then try out missing mpp handler
        #TODO: create handlers for different image types
        try:
            slide_mpp = handle_missing_mpp(slide)
        except:
            print(f"Error: couldn't load MPP from slide!")
            return None
    #steps = np.ceil(np.array(slide.dimensions)/(size*target_mpp/slide_mpp)).astype(int)
    #stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)
    #tile_target_size = np.array([size,size],dtype=int)
    #changed max amount of threads used
    with futures.ThreadPoolExecutor(cores) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(steps):  # row
            for j in range(steps):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_target_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an image as soon as they are loaded
        #im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        #for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps, desc='Reading WSI tiles', leave=False):
        im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        for tile_future in futures.as_completed(future_coords):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile
            #Image.fromarray(tile).save(f"{dir_name}/Tile_{j,i}.jpg")
        return im

    

def handle_missing_mpp(slide: openslide.OpenSlide) -> float:
    #print(f"Missing mpp in metadata of this file format, reading mpp from metadata!!")
    import xml.dom.minidom as minidom
    xml_path = slide.properties['tiff.ImageDescription']
    doc = minidom.parseString(xml_path)
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    #tile_size_px = um_per_tile / float(pixels[0].getAttribute("PhysicalSizeX"))
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    return mpp


def process_slide_svs(slide_url,slide_cores = 4):
    slide_name = Path(slide_url)
    slide_cache_dir = Path(f'{t_dir}/{str(slide_name).split("/")[-1].split(".")[0]}')
    #print(f"\nLoading {slide_name}")
    try:
        slide = openslide.OpenSlide(str(slide_url))
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        print(f"Unsupported format for {slide_name}")
        return None
    except Exception as e:
        print(f"Failed loading {slide_name}, error: {e}")
        return None
    if not exists(str(slide_cache_dir)):
        im=load_slide(slide,target_mpp=target_mpp, cores=slide_cores)
        if im is not None:
            slide_cache_dir.mkdir(parents=True, exist_ok=True)
            bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = im, patch_size = (size,size),step=size,cores=slide_cores)
            canny_img, _, _, _ = get_raw_tile_list(im.shape,bg_reject_array,rejected_tile_array,patch_shapes)
            get_patches(canny_img,str(slide_cache_dir))
        else:
            print(f"Error with slide  {slide_name}")
    else:
        print(f"Skipping {slide_name}...")
        

def save_img(file,svs=True):
    if svs:
        process_slide_svs(file)
    else:
        process_slide_jpg(file)

def main():
    t_dir.mkdir(parents=True, exist_ok=True)
    
    with mp.Pool(cores) as pool, tqdm(total=len(svs_dir)) as pbar:
        for _ in pool.imap_unordered(save_img, svs_dir):
            pbar.update(1)

if __name__ == "__main__":
    main()
