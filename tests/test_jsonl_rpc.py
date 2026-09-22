import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from common.jsonl import _dumps, read_jsonl
from common import jsonl_rpc as rpc


class RpcFrameTests(unittest.IsolatedAsyncioTestCase):
    def writer(self):
        return Mock(drain=AsyncMock(), wait_closed=AsyncMock())

    async def test_default_reader_reproduces_large_prompt_failure(self):
        reader = asyncio.StreamReader()
        reader.feed_data(_dumps({'text': 'x' * 70000}))
        reader.feed_eof()
        with self.assertRaises(ValueError):
            await read_jsonl(reader)

    async def test_large_request_and_response_survive_transport(self):
        payload = {'text': 'visual context ' * 10000}
        reader = asyncio.StreamReader(limit=rpc.RPC_MAX_MESSAGE_BYTES)
        reader.feed_data(_dumps({'method': 'echo', 'rpc_id': 'test', 'params': payload}))
        reader.feed_eof()
        writer = self.writer()
        await rpc._handle_client(reader, writer, {'echo': AsyncMock(return_value=payload)})
        reply = asyncio.StreamReader(limit=rpc.RPC_MAX_MESSAGE_BYTES)
        reply.feed_data(writer.write.call_args.args[0])
        reply.feed_eof()
        with patch.object(rpc.asyncio, 'open_connection', AsyncMock(return_value=(reply, self.writer()))) as connect:
            result = await rpc.send_request('localhost', 1, 'echo', payload)
        self.assertEqual(result['text'], payload['text'])
        self.assertEqual(connect.call_args.kwargs['limit'], rpc.RPC_MAX_MESSAGE_BYTES)

    async def test_oversized_request_returns_error_instead_of_empty_reply(self):
        reader = asyncio.StreamReader(limit=32)
        reader.feed_data(_dumps({'text': 'x' * 100}))
        reader.feed_eof()
        writer = self.writer()
        with patch.object(rpc, 'log_exception') as log:
            await rpc._handle_client(reader, writer, {})
        self.assertIn(b'Invalid or oversized RPC request', writer.write.call_args.args[0])
        log.assert_called_once()
        writer.close.assert_called_once()

    async def test_server_uses_same_frame_limit(self):
        with patch.object(rpc.asyncio, 'start_server', AsyncMock()) as start:
            await rpc.start_server('localhost', 1, {})
        self.assertEqual(start.call_args.kwargs['limit'], rpc.RPC_MAX_MESSAGE_BYTES)
