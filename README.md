# 스트리밍 알고리즘 비교 실험

MovieLens 1M 평점 스트림을 이용해 Bloom Filter와 Count-Min Sketch의 정확도, 메모리와 처리 속도를 비교하였습니다.

Python 3.10 이상의 환경이 필요하며 외부 패키지는 필요하지 않습니다.

```bash
python streaming_experiment.py
```

현재 폴더에 ml-1m.zip이 없으면 GroupLens에서 자동으로 다운로드합니다. 

```bash
python streaming_experiment.py path/to/ml-1m.zip
```

측정값은 JSON으로 터미널에 출력됩니다.