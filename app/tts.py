"""음성 안내(TTS) — Windows SAPI5(pyttsx3) 사용, 별도 스레드로 비차단 재생."""
import threading


def is_available() -> bool:
    """pyttsx3 사용 가능 여부."""
    try:
        import pyttsx3  # noqa: F401
        return True
    except Exception:
        return False


def speak_async(text: str):
    """텍스트를 백그라운드 스레드에서 읽어준다. 실패해도 조용히 무시."""
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass  # TTS 불가 환경에서도 앱 동작에 영향 없음

    threading.Thread(target=_run, daemon=True).start()
