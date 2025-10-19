from IPython.display import Audio
import numpy as np

# 8시간 동안 재생될 '무음(Silence)' 오디오를 생성합니다.
# 오디오가 재생되는 동안 Colab은 런타임 활동이 있다고 판단합니다.
# 28800초 (8시간) * 3000(샘플 속도) = 86,400,000개의 데이터 포인트
# 무료 세션의 12시간 한계에 가까운 시간을 시뮬레이션합니다.
audio_data = np.zeros(28800 * 3000, dtype=np.int8) 

Audio(audio_data, rate=3000, autoplay=True)
print("Session Keep Alive: Silence Audio is playing for up to 8 hours.")