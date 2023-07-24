from nehushtan.httpd.NehushtanHTTPRequestController import NehushtanHTTPRequestController

from gath.kit.GathDB import GathDB


class ApplicationController(NehushtanHTTPRequestController):

    def echo(self):
        self._reply_with_ok(None)

    def apply(self):
        """
        insert into gath.inn_application(
          `model`,
          `height`,`width`,
          `textual_inversion`,
          `lora`,  `lora_multiplier`,
          `prompt`,
          `negative_prompt`,
          `steps`,  `cfg`,  `scheduler`,
          `seed`,
          `status`,  `apply_time`
        )values(
            'sd-1.5-LeXiaoQi',
                512,512,
                null,
                null,1,
                'masterpiece, a LeXiaoQi with a computer',
                'bad hand, bad leg',
                20,7,'euler_a',
                null,
                'APPLIED',now()
        );
        """
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
        GathDB().register_one_task({

        })

    def list(self):
        pass
