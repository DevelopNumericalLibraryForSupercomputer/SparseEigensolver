import sys
import numpy as np

# .so 파일이 있는 디렉토리를 Python 경로에 추가
sys.path.append('/path/to/directory/containing/matvec.cpython-312-x86_64-linux-gnu.so')

try:
    import matvec
    print("Successfully imported matvec module")
    
    # 함수 테스트
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = matvec.matvec(arr)
    print("Result:", result)
except ImportError as e:
    print(f"Failed to import matvec module: {e}")
    print("sys.path:", sys.path)