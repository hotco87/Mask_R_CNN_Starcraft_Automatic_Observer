# 사전준비

#### 1) 옵저빙 데이터에 사용할 리플레이 데이터 선별

#### 2) 리플레이 데이터에 대한 raw data 만들기 (유닛, 지형 등등)
   => [replay_file].rep.raw 생성 (+ [replay_file].rep.vision, [replay_file].rep.terrain)  
   (ChaosLaucher => RELEASE 선택 => config => ai => bwapi-data/AI/RAWdataExtractor.dll)  
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)  

#### 3) 휴먼 데이터 수집      
   ==> [replay_file].rep.vpd 생성   
      (ChaosLaucher => "ai = bwapi-data/AI/ObserverModule.dll")  
      (생성위치 : C:\TM\starcraft\bwapi-data\write\)  


# 실행순서

1) 0_generate_compressed_npz.ipynb  
    : 딥러닝 입력으로 들어갈 인풋 만들기  
    : 사전준비 사항의 [replay_file].raw 와 [replay_file].rep.vpd 를 이용하여 딥러닝의 인풋으로 들어갈 채널 생성  
   
2) 1_replaydata_vpds_to_label_masked.py  
   : [replay_file].rep.vpd 를 이용하여 1x128x128 masked label 를 만드는 소스코드  
   --> 딥러닝의 아웃풋으로 사용됨  

3) 2_unzip_npz_label.py  
   : 1)의 input과 2)의 output 을 실제 딥러닝 input, ouput으로 들어갈 수 있도록 변경  
   
4) 3_main.py  
   : Masked R CNN 코드 실행  
   
5) 4_evaluated.ipynb  
   : 테스트 리플레이 데이터에 대해 vpx와 vpy를 만듦.  
   : test_data 폴더에서 작용함. ex) test_data, test_data2, test_data3 .. 
   
6) 5_generate_vpd.py
   : 5)에서 만든 vpx와 vpy를 이용하여 vpd를 만듦


### TODO
1) 여러 명의 리플레이 데이터에 대한 하나의 label을 생성
  : 10_multi_replay_generate_vpds_label_masked.ipynb <-- 여기에서 작업중