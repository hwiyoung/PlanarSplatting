#!/usr/bin/env python3
"""Download 3D BAG CityJSON tiles for specified scenes."""

import fiona
import gzip
import json
import os
import sys
import urllib.request
from shapely.geometry import shape, box


SCENES = {
    'amsterdam_jordaan': {
        'bbox': box(120500, 486500, 121000, 487000),
        'desc': 'Amsterdam Jordaan - historic canal houses, gable roofs',
    },
    'rotterdam_center': {
        'bbox': box(92000, 437000, 92500, 437500),
        'desc': 'Rotterdam center - modern commercial, flat roofs',
    },
    'delft_wijk': {
        'bbox': box(83500, 449500, 84000, 450000),
        'desc': 'Delft residential - Dutch row houses, mixed roofs',
    },
}

TILE_INDEX_URL = 'https://data.3dbag.nl/latest/tile_index.fgb'
OUT_DIR = 'results/stage3_synthetic_a/3dbag_raw'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Reading tile index...')
    sys.stdout.flush()
    with fiona.open(TILE_INDEX_URL) as src:
        tiles = list(src)
    print(f'Total tiles: {len(tiles)}')

    for scene_name, cfg in SCENES.items():
        print(f'\n=== {scene_name}: {cfg["desc"]} ===')
        sys.stdout.flush()

        matching = []
        for tile in tiles:
            tile_geom = shape(tile['geometry'])
            if tile_geom.intersects(cfg['bbox']):
                matching.append(tile)
        print(f'  Matching tiles: {len(matching)}')

        scene_dir = os.path.join(OUT_DIR, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        for ti, tile in enumerate(matching):
            tile_id = tile['properties']['tile_id']
            cj_url = tile['properties']['cj_download']
            safe_name = tile_id.replace('/', '-')
            json_path = os.path.join(scene_dir, f'{safe_name}.city.json')

            if os.path.exists(json_path):
                print(f'  [{ti+1}/{len(matching)}] {tile_id}: exists')
                continue

            print(f'  [{ti+1}/{len(matching)}] {tile_id}: downloading...',
                  end='', flush=True)
            try:
                gz_path = json_path + '.gz'
                urllib.request.urlretrieve(cj_url, gz_path)
                with gzip.open(gz_path, 'rb') as gz:
                    content = gz.read()
                with open(json_path, 'wb') as f:
                    f.write(content)
                os.remove(gz_path)
                data = json.loads(content)
                n_obj = len(data.get('CityObjects', {}))
                print(f' {len(content)//1024}KB, {n_obj} objects')
            except Exception as e:
                print(f' ERROR: {e}')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
