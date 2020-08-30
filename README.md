# 개념 변화 검출 엔진 (Concept Drift Detection Engine)

1. `Server.py` 를 `python3`에서 실행하는 것으로 서버의 실행이 가능합니다.
``python Server.py``

2. 서버 연결 정보는 `server_options.ini` 를 통해 수정 및 확인 할 수 있습니다.
    * 서버가 외부에서 정보를 받아오기 위한 IP 및 포트 등의 설정들이 여기 있습니다.
    * DB 및 Kafka 서버 보안 문제로 인해 자세한 정보는 공개하지 않으며, 샘플 정보만 포함되어 있습니다. 
    
3. `ConceptDriftDetectSystem` 폴더는 서버가 사용할 모델의 모체입니다.
이 폴더를 복제하는 방식으로 모델을 생성해나가기 때문에, 모델의 변경을 수행한다면 이 폴더 내의 코드를 변경해야할 것입니다.

4. 각 모델의 규모, 및 모델의 네트워크 입출력을 위한 설정의 탬플릿은 `ConceptDriftDetectSystem` 폴더의 `options.ini` 파일입니다.
    * 이 설정을 바꾸면, 이후에 생성되는 모든 모델의 설정에 영향을 미칠 수 있습니다.
    * 모델마다 변경되어야 할 수 있는 설정값(카프카 토픽, 유저명 등)은 모델 폴더의 복제 직후 `ManagerDriver.py`가 실행중에 작성합니다.