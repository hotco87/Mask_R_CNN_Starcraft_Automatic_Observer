# 사전준비

#### 1) 옵저빙 데이터에 사용할 리플레이 데이터 선별

#### 2) [replay_file].rep.raw 만들기 (RAWdataExtractor.dll 이용) // + [replay_file].rep.vision, [replay_file].rep.terrain
  => ChaosLaucher => RELEASE 선택 => config => ai => bwapi-data/AI/RAWdataExtractor.dll
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)   

#### 3) 휴먼 데이터 수집
   1) ChaosLaucher => "ai = bwapi-data/AI/ObserverModule.dll"
   2) 리플레이 관전 시키기
   3) [replay_file].rep.vpd 생성
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)


# 실행순서

1) 0_generate_compressed_npz.ipynb   : 딥러닝 입력으로 들어갈 인풋 만들기
    : [replay_file].raw 와 [replay_file].rep.vpd 를 이용하여 채널 생성
   --> 딥러닝의 인풋으로 사용됨.
   
2) 1_replaydata_vpds_to_label_masked.py
    : [replay_file].rep.vpd 를 이용하여 128x128 masked label 를 만드는 소스코드
   --> 딥러닝의 아웃풋으로 사용됨

3) 2_unzip_npz_label.py
   : 1)의 input과 2)의 output 을 실제 딥러닝 input, ouput으로 변경
   
4) 3_main.py
   : Masked R CNN 코드 실행
   
5) 4_evaluated.ipynb
   : 테스트 리플레이 데이터에 대해 vpx와 vpy를 만듦.
   : test_data 폴더에서 작용함. ex) test_data, test_data2, test_data3 .. 
   
6) 5_generate_vpd.py
   : 5)에서 만든 vpx와 vpy를 이용하여 vpd를 만듦
