import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


def timeit(name=None):
    # Decorator to print how long a function call took
    def decorator(fn):
        label = name or fn.__name__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000.0
            # print(f"[TIME] {label}: {dt:.2f} ms")
            return out
        return wrapper
    return decorator


def create_io_executor(max_workers=4):
    return ThreadPoolExecutor(max_workers=max_workers) #Creates a small background thread pool for handling I/O tasks


def submit_image_write(executor, pending_futures, cv2_module, path, image):
    # Queue image saving in a worker thread. Copy avoids data changing mid-write.
    pending_futures.append(executor.submit(cv2_module.imwrite, path, image.copy()))


def submit_xml_write(executor, pending_futures, writer_fn, path, boxes_xyxy_cls, width, height, image_path, class_names):
    pending_futures.append(
        executor.submit(
            writer_fn,
            path,
            list(boxes_xyxy_cls),
            width,
            height,
            image_path,
            class_names,
        )
    )


def wait_for_pending_writes(pending_futures):
    # Wait for all pending I/O tasks to finish, then clears the list of futures to free memory
    for future in pending_futures:
        future.result()
    pending_futures.clear() # Drop references to completed tasks
