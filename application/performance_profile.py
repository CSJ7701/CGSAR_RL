from pyinstrument import Profiler
import runpy

def profile_script(script_path):
    profiler = Profiler()
    profiler.start()
    runpy.run_path(script_path)
    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))

if __name__ == "__main__":
    script_path="wrapper.py"
    profile_script(script_path)
