from nehushtan.httpd.NehushtanHTTPRequestController import NehushtanHTTPRequestController


class ApplicationController(NehushtanHTTPRequestController):

    def echo(self):
        self._reply_with_ok(None)

    def apply(self):
        model = self._read_body(('model',))
        height = self._read_body(('height',))
        width = self._read_body(('width',))
        textual_inversion = self._read_body(('textual_inversion',))
        lora = self._read_body(('lora', 'ckpt'))
        lora_multiplier = self._read_body(('lora', 'multiplier'))
        prompt = self._read_body(('prompt',))
        negative_prompt = self._read_body(('negative_prompt',))
        steps = self._read_body(('steps',))
        cfg = self._read_body(('cfg',))
        scheduler = self._read_body(('scheduler',))
        seed = self._read_body(('seed',))

        # write into db

    def list(self):
        pass
