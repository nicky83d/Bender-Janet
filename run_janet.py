#!/usr/bin/env python3
from janet_core.controller import JanetController
from janet_web.app import create_app


def main():
    janet = JanetController()
    janet.start()
    app = create_app(janet)
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
