import socketserver
from typing import Tuple, Sequence, Union

from nehushtan.MessageQueue.implement.NehushtanMemoryMessageQueue import NehushtanMemoryMessageQueue
from nehushtan.httpd.NehushtanHTTPRequestFilter import NehushtanHTTPRequestFilter
from nehushtan.httpd.NehushtanHTTPRequestHandler import NehushtanHTTPRequestHandler
from nehushtan.httpd.NehushtanHTTPRouter import NehushtanHTTPRouter
from nehushtan.httpd.NehushtanHTTPService import NehushtanHTTPService
from nehushtan.httpd.implement.NehushtanHTTPRouteWithRestFul import NehushtanHTTPRouteWithRestFul

from gath import env


class GathInnRequestHandler(NehushtanHTTPRequestHandler):
    router = NehushtanHTTPRouter()
    mq = NehushtanMemoryMessageQueue()

    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer):
        # self.prepare_router()
        super().__init__(request, client_address, server)

    def seek_route_for_process_chains(self, method: str, path: str) \
            -> Tuple[Sequence[Union[type, str]], Tuple[Union[type, str], str]]:
        route = GathInnRequestHandler.router.check_request_for_route(method, path)
        # self.log_message(route.path_template)
        self.matched_arguments = route.matched_arguments
        self.matched_keyed_arguments = route.matched_keyed_arguments
        return route.get_filter_list(), route.get_controller_target()


class InnGateKeeper(NehushtanHTTPRequestFilter):

    def should_accept_request(self) -> bool:
        http_handler = self.get_http_handler()
        body: dict = http_handler.parsed_body_data
        token = body.get('token')
        print('InnGateKeeper::should_accept_request')
        print(token)
        if token == env.inn_token:
            return True
        else:
            return False


def start_server():
    GathInnRequestHandler.router.register_route(
        NehushtanHTTPRouteWithRestFul(
            '/api',
            'gath.inn.server.api',
            [InnGateKeeper]
        )
    )
    NehushtanHTTPService.run_with_threading_server(GathInnRequestHandler, 4466)


if __name__ == '__main__':
    start_server()
