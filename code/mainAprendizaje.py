# -*- coding: utf-8 -*-
from code.modules import loadRegions
from code.modules import learning_characterization as LC

regions = loadRegions.get_regions_from_csv('aprendizaje/audiosAprendizaje.csv')
LC.generate_MFFC_LPC_region_list(regions, True)
LC.generate_GMM_region_list( regions , 'directo' )
LC.generate_GMM_region_list( regions , 'lpc' )

