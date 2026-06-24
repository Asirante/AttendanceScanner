"""음성 안내(TTS) — Windows SAPI5(pyttsx3) 사용, 별도 스레드로 비차단 재생.
TTS 실패 시 윈도우 기본 비프음으로 대체합니다.
"""

import threading


def is_available() -> bool:
    """pyttsx3 사용 가능 여부."""
    try:
        import pyttsx3  # noqa: F401

        return True
    except Exception:
        return False


def speak_async(text: str):
    """텍스트를 백그라운드 스레드에서 읽어준다. 오류 발생 시 비프음 재생."""

    def _run():
        tts_success = False
        try:
            import pyttsx3
            import pythoncom

            # 윈도우 환경 스레드에서 COM(SAPI5) 객체를 쓰기 위한 초기화
            pythoncom.CoInitialize()

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

            tts_success = (
                True  # 정상적으로 끝까지 재생되었음을 표시
            )

        except Exception as e:
            # 에러 로그 — print 자체가 실패해도(예: stdout 이상) 앱엔 영향 없게 방어
            try:
                print(
                    f"[TTS 오류] 음성 출력 실패, 비프음으로 대체합니다: {e}"
                )
            except Exception:
                pass
        finally:
            # 사용이 끝난 후 COM 자원 해제 (메모리 누수 및 충돌 방지)
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass

            # TTS 재생에 실패했다면 비프음(Beep)으로 알림
            if not tts_success:
                try:
                    import winsound  # Windows 전용 — 다른 모듈들과 동일하게 지연 import

                    # 800Hz 주파수로 300ms(0.3초) 동안 비프음 발생
                    winsound.Beep(800, 300)
                except Exception:
                    pass

    # 데몬 스레드로 실행하여 메인 프로그램 종료 시 함께 종료되도록 보장
    threading.Thread(target=_run, daemon=True).start()
