"""生成一个微通道测试 stk 文件 (用绝对路径)"""
import os
d = '/mnt/d/aicooling/3d-ice'

stk = f'''material SILICON :
   thermal conductivity     1.30e-4 ;
   volumetric heat capacity 1.628e-12 ;

material TOY :
   thermal conductivity     1.0e-7 ;
   volumetric heat capacity 1.628e-12 ;

top pluggable heat sink :
   spreader length 1000, width 1000, height 100 ;
   material TOY ;
   plugin "{d}/heatsink_plugin/loaders/python/python_loader.so", "{d}/heatsink_microchannel.py 50 100 100 10 293" ;

bottom heat sink :
   heat transfer coefficient 5.0e-10 ;
   temperature               293 ;

dimensions :
   chip length 1000, width 1000 ;
   cell length  50, width  50 ;
   non-uniform false;

die TOP_IC :
   source  5 SILICON ;
   layer  495 SILICON ;

stack:
   die     MY_DIE     TOP_IC    floorplan "../data/case_microchannel_0.flp" ;

solver:
   steady ;
   initial temperature 293 ;
   numofcores 1 ;

output:
   Tmap  ( MY_DIE, "../data/case_microchannel_test_temp.txt", final ) ;
'''

with open(os.path.join(d, 'bin', 'case_micro_abs.stk'), 'w') as f:
    f.write(stk)
print('Written: case_micro_abs.stk')
print('Now run: ./3D-ICE-Emulator case_micro_abs.stk')
