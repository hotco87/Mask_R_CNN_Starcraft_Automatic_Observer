# 실행순서

#### 0_generate_compressed_npz.ipynb  
    : 딥러닝 입력으로 들어갈 인풋 만들기  
    : 사전준비 사항의 [replay_file].raw 와 [replay_file].rep.vpd 를 이용하여 딥러닝의 인풋으로 들어갈 채널 생성 
    : 'C:/TM/starcraft/bwapi-data/write/raw/' <- 이 폴더에 하나의 리플레이당 (12.rep.raw, 12.rep.terrain, 12.rep.vision), 12.rep.vpd 가 있으면 됨.  
    : https://drive.google.com/file/d/1wLdA7p6uB7rzUO2GK6_M3E_hP0VML9sp/view?usp=sharing  
      => 'C:/TM/starcraft/bwapi-data/write/raw/' 이 폴더에 위 링크 압축 풀어서 넣으면 됨
    : data_compressed3/[replay_file]/   <- 이 폴더에 npz, vpd 생성  

#### 1_replaydata_vpds_to_label_masked.py  
    : [replay_file].rep.vpd 를 이용하여 1x128x128 masked label 를 만드는 소스코드
    : data_compressed3/[replay_file]/   <- 이 폴더에 정답 masked label 생성 (vpds_label_masked.npy)
      => 딥러닝의 아웃풋으로 사용됨  

#### 2_unzip_npz_label.py  
    : 1)의 input과 2)의 output 을 실제 딥러닝 input, ouput 으로 들어갈 수 있도록 변경  (training/testing 구분)
    : training_data/[replay_file]/   <- 이 폴더에 ".npy" 파일들이 있으며 인풋과 아웃풋이 합쳐져 있음
    : testing_data/[replay_file]/   <- 이 폴더에 ".npy" 파일들이 있으며 인풋과 아웃풋이 합쳐져 있음
   
#### 3_main.py  
    : #### https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html    
      => 먼저 이 코드가 돌아가게끔 환경세팅을 잡아줘야함. (torchvision 설치 등)
      => anaconda3\envs\starcraft\Lib\site-packages\torchvision\models\detection\transform.py 에서 image = self.normalize(image) <- 이부분 주석처리
    : Masked R CNN 코드 실행  
   
#### 4_evaluated.ipynb  
    : (testing_data) 테스트 리플레이 데이터에 대해 vpx와 vpy를 만듦.
    : 생성된 vpx와 vpy를 이용하여 starcraft에서 확인해보기 위해 vpd를 만듦

# 사전준비 2

#### 1) 옵저빙 데이터에 사용할 리플레이 데이터 선별

#### 2) 리플레이 데이터에 대한 raw data 만들기 (유닛, 지형 등등)
   => [replay_file].rep.raw 생성 (+ [replay_file].rep.vision, [replay_file].rep.terrain)  
   (ChaosLaucher => RELEASE 선택 => config => ai => bwapi-data/AI/RAWdataExtractor.dll)  
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)  

#### 3) 휴먼 데이터 수집  
   ==> [replay_file].rep.vpd 생성   
      (ChaosLaucher => "ai = bwapi-data/AI/ObserverModule.dll")  
      (생성위치 : C:\TM\starcraft\bwapi-data\write\)  



### TODO
1) 여러 명의 리플레이 데이터에 대한 하나의 label을 생성  
  : 10_multi_replay_generate_vpds_label_masked.ipynb <-- 여기에서 작업중