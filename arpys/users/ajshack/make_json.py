import json

def add_hv_scan_data(data_hv_scan, json_file='spectra_info.json'):
 with open(json_file, 'r') as f:
  data = json.load(f)

 if 'hv_scan' not in data:
  data['hv_scan'] = {}

 # Add new hv_scan data
 for material, files in data_hv_scan['hv_scan'].items():
  if material not in data['hv_scan']:
   data['hv_scan'][material] = {}
  for filepath, attrs in files.items():
   data['hv_scan'][material][filepath] = attrs

 with open(json_file, 'w') as f:
  json.dump(data, f, indent=4)


def update_filepaths(json_file_path, old_base_path, new_base_path):
 with open(json_file_path, 'r') as file:
  data = json.load(file)

 def update_path(d):
  if isinstance(d, dict):
   return {k if not k.startswith(old_base_path) else k.replace(old_base_path, new_base_path): update_path(v) for k, v in
           d.items()}
  return d

 updated_data = update_path(data)

 with open(json_file_path, 'w') as file:
  json.dump(updated_data, file, indent=4)


data = {"fermi_map":
         {"co33tas2":
           {"C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0012.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0025.h5":
             {"alignment": "GM", "polarization": "CR", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0026.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0020.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 86,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0021.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 86,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0022.h5":
             {"alignment": "GM", "polarization": "CR", "photon_energy": 86,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0023.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 86,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0024.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 83,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0035.h5":
             {"alignment": "GM", "polarization": "CR", "photon_energy": 83,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0034.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 83,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0027.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 90,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0036.h5":
             {"alignment": "GM", "polarization": "CR", "photon_energy": 90,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0038.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 90,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0039.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 90,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0028.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 96,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0029.h5":
             {"alignment": "GM", "polarization": "CR", "photon_energy": 96,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0030.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 96,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april2024/Co13TaS2_0031.h5":
             {"alignment": "GM", "polarization": "CL", "photon_energy": 96,
              "k_conv": False, "masked": False, "dewarped": False}},
          "co55tas2":
           {"C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0010.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 150,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0019.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0020.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 92,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0024.h5":
             {"alignment": "GM", "polarization": "LH", "photon_energy": 115,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0035.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0022.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 192,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0023.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 187,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0024.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 182,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0025.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 176,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0026.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 162,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0029.h5":
             {"alignment": "GK", "polarization": "LV", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0030.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 76,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0037.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 187,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0038.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 173,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0039.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 159,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0040.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 146,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0041.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 121,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0042.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 98,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0043.h5":
             {"alignment": "GK", "polarization": "LV", "photon_energy": 146,
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_113022/CoTS_0021.h5":
             {"alignment": "GK", "polarization": "LH", "photon_energy": 192,
              "k_conv": False, "masked": False, "dewarped": False}
            }
          }
        }

data_hv_scan = \
{"hv_scan":
         {"co33tas2":
           {"C:/Users/proxi/Documents/coding/data/ssrl_april20224/Co13TaS2_0032.h5":
             {"alignment": "GM", "polarization": "CR", "hv" : [64, 99],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april20224/Co13TaS2_0033.h5":
             {"alignment": "GM", "polarization": "CL", "hv" : [64, 99],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april20224/Co13TaS2_0017.h5":
             {"alignment": "GM", "polarization": "LH", "hv" : [60, 72],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april20224/Co13TaS2_0018.h5":
             {"alignment": "GM", "polarization": "LH", "hv" : [72, 121],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_april20224/Co13TaS2_0019.h5":
             {"alignment": "GM", "polarization": "LH", "hv" : [121, 181],
              "k_conv": False, "masked": False, "dewarped": False}},
          "co55tas2":
           {"C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0017.h5":
             {"alignment": "GM", "polarization": "LH", "hv" : [70, 120],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0018.h5":
             {"alignment": "GM", "polarization": "LH", "hv" : [30, 70],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/ssrl_071522/Co-TaS2_0036.h5":
             {"alignment": "GK", "polarization": "LH", "hv" : [30, 170],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/APS/cotas/kz_scan/hv_290_600":
             {"alignment": "GM", "polarization": "LH", "hv" : [290, 600],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/APS/cotas/kz_scan/hv_400_540":
             {"alignment": "GM", "polarization": "LH", "hv" : [400, 540],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/APS/cotas/kz_scan/hv_440_540":
             {"alignment": "GM", "polarization": "LV", "hv" : [440, 540],
              "k_conv": False, "masked": False, "dewarped": False},
            "C:/Users/proxi/Documents/coding/data/APS/cotas/kz_scan/hv_450_740":
             {"alignment": "GM", "polarization": "LV", "hv" : [450, 740],
              "k_conv": False, "masked": False, "dewarped": False}
            }
          }
        }

#with open('spectra_info.json', 'w') as json_file:
#    json.dump(data, json_file, indent=4)

#add_hv_scan_data(data_hv_scan, json_file='spectra_info.json')

# Paths for the json file and the old and new base paths
json_file_path = 'C:/Users/proxi/Documents/coding/arpys/arpys/users/ajshack/spectra_info.json'
old_base_path = 'C:/Users/proxi/Documents/coding/data'
new_base_path = 'C:/Users/proxi/Documents/coding/arpes_data'

# Update the filepaths
update_filepaths(json_file_path, old_base_path, new_base_path)
