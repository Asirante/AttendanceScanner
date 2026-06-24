"""GUI 앱 실행 진입점.  실행:  python run_app.py"""

import os
import sys

# --noconsole(windowed) 빌드에서는 sys.stdout/stderr가 None이라
# 어딘가에서 print()나 .write()를 호출하면 크래시함. 더미로 채워서 방지.
# (이 패치는 app.gui를 import하기 전, 파일의 가장 위에서 실행되어야 함)
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

from app.gui import main

if __name__ == "__main__":
    main()
