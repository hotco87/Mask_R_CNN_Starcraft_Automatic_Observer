# 사전준비

### 1) 옵저빙 데이터에 사용할 리플레이 데이터 선별

### 2) [replay_file].rep.raw, [replay_file].rep.vision, [replay_file].rep.terrain 만들기 (RAWdataExtractor.dll 이용)
  => ChaosLaucher => RELEASE 선택 => config => ai => bwapi-data/AI/RAWdataExtractor.dll
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)   

### 3) 휴먼 데이터 수집
   1) ChaosLaucher => "ai = bwapi-data/AI/ObserverModule.dll"
   2) 리플레이 관전 시키기
   3) [replay_file].rep.vpd 생성
   (생성위치 : C:\TM\starcraft\bwapi-data\write\)

### 4) 딥러닝 입력으로 들어갈 인풋 만들기
   1) [replay_file].raw 와 [replay_file].rep.vpd 를 이용하여 채널 생성


# 