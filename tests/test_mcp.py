"""Real subprocess/stdio handshake, tool discovery, replay parity, and process isolation."""
import asyncio
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from marketcanvas.scenarios import build_reference

ROOT=Path(__file__).resolve().parents[1]


def test_real_mcp_transport_matches_python():
    async def run():
        params=StdioServerParameters(command=sys.executable,args=["-m","marketcanvas.server"],cwd=str(ROOT))
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as client:
                await client.initialize()
                tools=await client.list_tools()
                assert {t.name for t in tools.tools}=={"get_canvas_state","get_action_schema","execute_action","get_current_reward","reset_environment"}
                schema=await client.call_tool("get_action_schema",{})
                assert not schema.isError
                reference=build_reference()
                for transition in reference.trajectory:
                    result=await client.call_tool("execute_action",{"action":transition["action"]})
                    assert not result.isError
                    assert result.structuredContent["observation"]==transition["next_observation"]
                    assert result.structuredContent["reward"]==0
                preview=await client.call_tool("get_current_reward",{})
                assert preview.structuredContent["score"]==1
                bad=await client.call_tool("execute_action",{"action":{"op":"delete_element","id":"missing"}})
                assert bad.structuredContent["info"]["error"]
                done=await client.call_tool("execute_action",{"action":{"op":"submit"}})
                assert done.structuredContent["reward"]==1
                repeat=await client.call_tool("execute_action",{"action":{"op":"submit"}})
                assert repeat.isError
                # A second process starts blank, independently of the completed first episode.
                async with stdio_client(params) as (read2,write2):
                    async with ClientSession(read2,write2) as other:
                        await other.initialize()
                        blank=await other.call_tool("get_canvas_state",{})
                        assert blank.structuredContent["elements"]==[]
                reset=await client.call_tool("reset_environment",{"seed":9,"task":{"headline":"Winter Sale"}})
                assert reset.structuredContent["observation"]["task"]["headline"]=="Winter Sale"
    asyncio.run(run())
