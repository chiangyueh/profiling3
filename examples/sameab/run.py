import os
import resource

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))

import ctypes
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path.insert(0, str(ROOT))

from clean.predictor import predict
from clean.runtime.graphs import engine as graph_builder
from clean.runtime.native import graph as native_graph


def location(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class Values:
    def __init__(self, stream):
        self.stream = stream
        self.arrays = {}
        self.states = {}
        self.objects = {}
        self.references = []

    def write(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def encode(self, value, seen=None):
        if seen is None:
            seen = set()
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, np.generic):
            return self.encode(value.item(), seen)
        if isinstance(value, float):
            return value if math.isfinite(value) else {"float": str(value)}
        if isinstance(value, Path):
            return {"path": location(value)}
        if isinstance(value, np.ndarray):
            key = (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).digest())
            if key not in self.arrays:
                identifier = "A%05d" % (len(self.arrays) + 1)
                self.arrays[key] = identifier
                self.write({"id": identifier, "type": "array", "dtype": str(value.dtype),
                            "shape": list(value.shape), "data": value.tolist()})
            return {"array": self.arrays[key], "dtype": str(value.dtype), "shape": list(value.shape)}
        if isinstance(value, types.ModuleType):
            return {"module": value.__name__, "file": location(value.__file__) if value.__file__ else None}
        if isinstance(value, ctypes.CDLL):
            symbols = {}
            for name, function in vars(value).items():
                if isinstance(function, ctypes._CFuncPtr):
                    symbols[name] = {"argtypes": [item.__name__ for item in function.argtypes or ()],
                                     "restype": getattr(function.restype, "__name__", str(function.restype))}
            return {"library": location(value._name), "symbols": symbols}
        if isinstance(value, subprocess.CompletedProcess):
            return {"args": self.encode(value.args, seen), "returncode": value.returncode,
                    "stdout": value.stdout, "stderr": value.stderr}
        if isinstance(value, dict):
            if all(isinstance(key, str) for key in value):
                return {key: self.encode(item, seen) for key, item in value.items()}
            return {"dict_items": [[self.encode(key, seen), self.encode(item, seen)] for key, item in value.items()]}
        if isinstance(value, list):
            return [self.encode(item, seen) for item in value]
        if isinstance(value, tuple):
            return {"tuple": [self.encode(item, seen) for item in value]}
        if isinstance(value, range):
            return {"range": {"start": value.start, "stop": value.stop, "step": value.step},
                    "items": list(value)}
        if isinstance(value, set):
            return {"set": [self.encode(item, seen) for item in value]}
        if type(value).__module__.startswith("clean.") and hasattr(value, "__dict__"):
            address = id(value)
            if address not in self.objects:
                self.objects[address] = "O%04d" % (len(self.objects) + 1)
                self.references.append(value)
            identifier = self.objects[address]
            if address in seen:
                return {"object_cycle": identifier}
            fields = self.encode(vars(value), seen | {address})
            record = {"object": identifier, "class": type(value).__module__ + "." + type(value).__name__, "fields": fields}
            key = json.dumps(record, ensure_ascii=False, allow_nan=False)
            if key not in self.states:
                state = "S%05d" % (len(self.states) + 1)
                self.states[key] = state
                self.write({"id": state, "type": "object_state", **record})
            return {"object": identifier, "state": self.states[key], "class": record["class"]}
        raise TypeError("unrecorded input/output type: " + type(value).__module__ + "." + type(value).__name__)


class Calls:
    def __init__(self, values):
        self.values = values
        self.records = []
        self.stack = []
        self.frames = {}
        self.event = 0

    def begin(self, name, file, line, inputs, kind="python", globals_=None, closure=None):
        self.event += 1
        record = {"id": len(self.records) + 1, "parent": self.stack[-1]["id"] if self.stack else None,
                  "depth": len(self.stack), "call_event": self.event, "function": name,
                  "source": file, "line": line, "kind": kind, "inputs": self.values.encode(inputs)}
        if globals_:
            record["globals"] = self.values.encode(globals_)
        if closure:
            record["closure"] = self.values.encode(closure)
        self.records.append(record)
        self.stack.append(record)
        return record

    def end(self, record, output, self_after=None):
        assert self.stack[-1] is record
        self.event += 1
        record["return_event"] = self.event
        record["output"] = self.values.encode(output)
        if self_after is not None:
            record["self_after"] = self.values.encode(self_after)
        self.stack.pop()

    def profile(self, frame, event, argument):
        code = frame.f_code
        if not code.co_filename.startswith(str(ROOT / "clean") + "/") or code.co_name.startswith("<"):
            return
        if event == "call":
            count = code.co_argcount + code.co_kwonlyargcount
            names = list(code.co_varnames[:count])
            if code.co_flags & inspect.CO_VARARGS:
                names.append(code.co_varnames[count])
                count += 1
            if code.co_flags & inspect.CO_VARKEYWORDS:
                names.append(code.co_varnames[count])
            inputs = {name: frame.f_locals[name] for name in names}
            instance = frame.f_locals.get("self")
            name = frame.f_globals["__name__"] + "."
            if instance is not None:
                name += type(instance).__name__ + "."
            name += code.co_name
            globals_ = {name: frame.f_globals[name] for name in dict.fromkeys(code.co_names)
                        if name in frame.f_globals and not callable(frame.f_globals[name])
                        and not isinstance(frame.f_globals[name], types.ModuleType)}
            closure = {name: frame.f_locals[name] for name in code.co_freevars if name in frame.f_locals}
            self.frames[id(frame)] = self.begin(name, location(code.co_filename), code.co_firstlineno, inputs,
                                                 globals_=globals_, closure=closure)
        elif event == "return":
            record = self.frames.pop(id(frame))
            self.end(record, argument, frame.f_locals.get("self"))


def array(pointer, count, dtype, shape=None):
    if not pointer:
        return None
    ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    values = np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctype)), shape=(count,)).copy()
    return values.reshape(shape) if shape else values


class Native:
    def __init__(self, library, calls):
        self.library = library
        self.calls = calls

    def __getattr__(self, name):
        function = getattr(self.library, name)

        def invoke(*args):
            if name == "fa_latency_order":
                n, a, b, output = args
                inputs = {"n": n, "a": array(a, n, "int32"), "b": array(b, n, "int32")}
                file = ROOT / "clean/runtime/native/fa_latency_eval.cpp"
            elif name == "fa_compact_graph":
                n, root, a, b, labels, zero, out_a, out_b, out_labels, out_root = args
                count = len(inspect.currentframe().f_back.f_locals["zero"])
                inputs = {"n": n, "root": root, "a": array(a, n, "int32"), "b": array(b, n, "int32"),
                          "labels": array(labels, n, "int32"), "zero": array(zero, count, "uint8")}
                file = ROOT / "clean/runtime/native/fa_branch_graph.cpp"
            elif name == "fa_latency_eval":
                n, templates, dimensions, root, a, b, labels, matrix, rates, gradient = args
                inputs = {"n": n, "templates": templates, "dimensions": dimensions, "root": root,
                          "a": array(a, n, "int32"), "b": array(b, n, "int32"), "which": array(labels, n, "int32"),
                          "matrix": array(matrix, templates * dimensions, "float64", (templates, dimensions)),
                          "rates": array(rates, dimensions, "float64"), "gradient_requested": bool(gradient)}
                file = ROOT / "clean/runtime/native/fa_latency_eval.cpp"
            else:
                raise ValueError("unrecorded native function: " + name)
            line = next(index for index, text in enumerate(file.read_text().splitlines(), 1) if name + "(" in text)
            record = self.calls.begin(name, location(file), line, inputs, "native")
            result = function(*args)
            if name == "fa_latency_order":
                output = {"return": result, "order": array(output, result, "int32")}
            elif name == "fa_compact_graph":
                output = {"return": result, "a": array(out_a, result, "int32"), "b": array(out_b, result, "int32"),
                          "labels": array(out_labels, result, "int32"),
                          "root": ctypes.cast(out_root, ctypes.POINTER(ctypes.c_int)).contents.value}
            else:
                output = {"return": result, "gradient": array(gradient, dimensions, "float64")}
            self.calls.end(record, output)
            return result

        return invoke


def compare_measurement(inputs, result, records):
    source = ROOT / "calibration/measurements.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        matches = [(line, row) for line, row in enumerate(reader, 2)
                   if all(row.get(key) == str(value) for key, value in inputs.items())]
    if len(matches) != 1:
        raise ValueError("expected exactly one matching measured input, found %d" % len(matches))
    line, row = matches[0]
    if row["cost_class"].split("|")[0] != "same_ab":
        raise ValueError("the measured record is not a SameAB case")
    predicted_class = next(record["output"] for record in records
                           if record["function"] == "clean.runtime.schedules.cost_class")
    if predicted_class != row["cost_class"]:
        raise ValueError("predicted and measured execution classes differ")
    measured = float(row["duration_us"])
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("invalid measured latency")
    with (FOLDER / "measurement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    difference = result["predicted_us"] - measured
    comparison = {
        "source": location(source), "source_line": line, "case": row["case"],
        "measurement_snapshot": "measurement.csv", "measurement_scope": row["measurement_scope"],
        "cost_class": predicted_class,
        "measured_us": measured, "predicted_us": result["predicted_us"],
        "signed_error_us": difference, "signed_error_pct": difference / measured * 100,
        "absolute_error_pct": abs(difference) / measured * 100,
        "evaluation_scope": "existing_measurement_comparison",
        "limitation": "既有校準資料對照，不是新 shape 的獨立泛化驗證；本次沒有重新量測 NPU。",
    }
    (FOLDER / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    return comparison


def save_process(inputs, result, comparison, records, values):
    with (FOLDER / "process.txt").open("w") as stream:
        stream.write("SameAB：clean.predictor.predict() 的一次實際 CPU 預測呼叫\n")
        stream.write("predicted_us 是模型預估的 NPU latency；本次沒有執行 NPU。\n")
        stream.write("各階段數值是模型關鍵路徑的時間貢獻，不是獨立量測的階段耗時。\n")
        stream.write("記錄全部 clean 具名 Python 函式呼叫，以及原生 C++／SDK helper 邊界的輸入輸出。\n")
        stream.write("陣列與物件狀態完整存於 values.jsonl，A/S 編號是引用位置；沒有截斷陣列。\n")
        stream.write("call_event／return_event 表示實際先後順序；parent 表示呼叫它的函式編號。\n")
        stream.write("輸入：" + json.dumps(inputs, ensure_ascii=False) + "\n")
        stream.write("結果：" + json.dumps(result, ensure_ascii=False) + "\n")
        stream.write("實測來源：%s:%d，case=%s；原始一筆資料完整保存在 measurement.csv。\n" %
                     (comparison["source"], comparison["source_line"], comparison["case"]))
        stream.write("預測與量測記錄的入參、dtype、layout、drop mask 和 cost_class 完全對應：" +
                     comparison["cost_class"] + "\n")
        stream.write("實測 %.9f µs；預測 %.9f µs；預測減實測 %+.9f µs；絕對百分比誤差 %.6f%%。\n" %
                     (comparison["measured_us"], comparison["predicted_us"], comparison["signed_error_us"],
                      comparison["absolute_error_pct"]))
        stream.write("預測完成後才讀取實測資料做對照，實測值沒有傳入 predict()。\n")
        stream.write(comparison["limitation"] + "\n")
        stream.write("主流程：predict → load_model → workloads.build → graphs.build → costs.rates → timing.evaluate → 返回結果\n")
        stream.write("呼叫數：" + str(len(records)) + "；陣列數：" + str(len(values.arrays)) + "；物件狀態數：" + str(len(values.states)) + "\n\n")
        stream.write("主流程索引（下方相同編號有完整輸入與輸出）：\n")
        steps = (
            ("讀取模型係數與硬體設定", "clean.runtime.costs.load_model"),
            ("檢查及正規化入參", "clean.runtime.inputs.normalize"),
            ("選擇 SameAB 排程與切塊", "clean.runtime.schedules.select"),
            ("組成兩個 BMM 的 Host API 請求", "clean.runtime.native.matmul.mm_requests"),
            ("呼叫 CANN Host API 得到兩個 BMM 的幾何參數", "clean.runtime.native.matmul.geometry"),
            ("產生本分支的工作量與任務", "clean.runtime.schedules.sameab_schedule.workload"),
            ("建立並壓縮資源與依賴關係圖", "clean.runtime.graphs.build"),
            ("將成本係數對應到圖的特徵欄位", "clean.runtime.costs.rates"),
            ("計算關鍵路徑總時間與階段貢獻", "clean.runtime.timing.evaluate"),
            ("回傳預測結果", "clean.predictor.predict"),
        )
        for label, function in steps:
            record = next(item for item in records if item["function"] == function)
            stream.write("[%04d] %s → %s @ %s:%d\n" %
                         (record["id"], label, function, record["source"], record["line"]))
        stream.write("\n完整呼叫記錄（依進入函式的順序列出，回傳先後見 return_event）：\n\n")
        for record in records:
            stream.write("[%04d] %s @ %s:%d\n" % (record["id"], record["function"], record["source"], record["line"]))
            stream.write("parent=%s kind=%s call_event=%s return_event=%s\n" %
                         (record["parent"], record["kind"], record["call_event"], record["return_event"]))
            for field, label in (("inputs", "輸入"), ("globals", "使用的全域常數"), ("closure", "閉包參數"),
                                 ("output", "輸出"), ("self_after", "呼叫後物件狀態")):
                if field in record:
                    stream.write(label + "：" + json.dumps(record[field], ensure_ascii=False) + "\n")
            if record is not records[-1]:
                stream.write("\n")


def main():
    inputs = json.loads((FOLDER / "input.json").read_text())
    original_graph = native_graph.graph_engine
    original_compact = native_graph.compact_engine
    original_builder = graph_builder.graph_engine
    original_run = subprocess.run
    with (FOLDER / "values.jsonl").open("w") as stream:
        values = Values(stream)
        calls = Calls(values)
        proxies = {}

        def proxy(library):
            if id(library) not in proxies:
                proxies[id(library)] = Native(library, calls)
            return proxies[id(library)]

        def graph_engine():
            return proxy(original_graph())

        def compact_engine():
            return proxy(original_compact())

        def run(*args, **kwargs):
            caller = inspect.currentframe().f_back
            record = calls.begin("subprocess.run", location(caller.f_code.co_filename), caller.f_lineno,
                                 {"args": args, "kwargs": kwargs}, "native_process")
            command = args[0] if args else kwargs["args"]
            if Path(command[0]).name == "matmul_tiling":
                record["native_source"] = "clean/runtime/native/fa_branch_matmul_tiling.cpp"
            else:
                record["native_sources"] = [location(item) for item in command if str(item).endswith(".cpp")]
            output = original_run(*args, **kwargs)
            calls.end(record, output)
            return output

        native_graph.graph_engine = graph_builder.graph_engine = graph_engine
        native_graph.compact_engine = compact_engine
        subprocess.run = run
        try:
            sys.setprofile(calls.profile)
            result = predict(inputs)
        finally:
            sys.setprofile(None)
            native_graph.graph_engine = original_graph
            native_graph.compact_engine = original_compact
            graph_builder.graph_engine = original_builder
            subprocess.run = original_run
        if result["implementation"] != "same_ab":
            raise ValueError("the example did not select SameAB")
        assert not calls.stack and not calls.frames
        with (FOLDER / "calls.jsonl").open("w") as output:
            for record in calls.records:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        (FOLDER / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        comparison = compare_measurement(inputs, result, calls.records)
        save_process(inputs, result, comparison, calls.records, values)
    print("EXAMPLE=" + str(FOLDER))
    print("predicted_us=%.9f" % result["predicted_us"])
    print("measured_us=%.9f" % comparison["measured_us"])
    print("absolute_error_pct=%.6f" % comparison["absolute_error_pct"])
    print("CALLS=" + str(len(calls.records)))


if __name__ == "__main__":
    main()
