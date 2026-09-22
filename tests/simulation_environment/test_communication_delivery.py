# Copyright (c) 2025 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pin the message delivery contract of the communication link.

`get_messages` prunes delivered items with `list(set(memory) - set(ready))`, so
the order of the returned list is an artefact of object identity and varies
between runs. The verification harness normalises that one list before diffing
(`reproduction/phase1/_common.normalize_known_noise`). These tests pin what must
stay stable underneath that normalisation -- *which* messages are delivered and
*on which tick* -- and deliberately do not assert delivery order. Once the
pruning stops going through a set, add an ordering test here and delete the
normalisation from the harness.
"""

import pytest

import source.simulation_environment.communication.communication as communication_module
from source.simulation_environment.communication.communication import Communication
from source.simulation_environment.communication.fault_models.fault_logic_base import ComEvent
from source.simulation_environment.configuration_error import SimulationConfigurationError
from source.simulation_environment.obps.message import SensorMsg


DELAY_MS = 100


def _cfg(delay=DELAY_MS, jitter=0, drop_rate=0.0, offline=0, trigger=10**9):
    """A communication block with a deterministic network and a dormant fault model."""
    return {
        "network": {
            "type": "uniform",
            "packet_drop_rate": drop_rate,
            "delay": delay,
            "jitter": jitter,
        },
        "fault": {"type": "sit", "offline": offline, "trigger": trigger},
    }


def _msg(msg_id):
    return SensorMsg(sender="sensor", msg_id=msg_id, time_stamp=0, content=None)


class _RecordingRedis:
    """Stand-in for `logs.redis_data`.

    `save_stream_com` snapshots the log item the way the real sink does: it
    serialises synchronously, before `Communication.log` clears the buffer. A
    sink that instead retained the list would observe it empty, because
    `ComLogItem` stores the caller's list by reference.
    """

    K_COM = "COM"

    def __init__(self):
        self.entries = []

    def get_h_key(self, h_set, key):
        return b"COM:0"

    def get_stream_name(self, exp_name, key):
        return "{0}:{1}".format(exp_name, key)

    def save_stream_com(self, stream, item):
        self.entries.append(
            {"t": item.t, "online": item.online, "dropped": list(item.dropped_packets)}
        )


@pytest.fixture
def sink(monkeypatch):
    recorder = _RecordingRedis()
    monkeypatch.setattr(communication_module, "rd", recorder)
    return recorder


@pytest.fixture
def link(sink):
    return Communication(0.05, "exp", _cfg())


class _StubFault:
    """A fault model whose event sequence is fixed by the test."""

    def __init__(self, events, clock_probe=None):
        self.events = list(events)
        self.seen_times = []
        self.seen_clocks = []
        self._clock_probe = clock_probe

    def step(self, sim_time):
        self.seen_times.append(sim_time)
        if self._clock_probe is not None:
            self.seen_clocks.append(self._clock_probe())
        return self.events.pop(0) if self.events else ComEvent.ONLINE


class TestIngressAndAging:

    def test_message_is_queued_with_the_network_delay_as_its_age(self, link):
        link.ingress(sim_time=0, msg=_msg(1))

        item = link.memory[0][0]
        assert item.age == DELAY_MS
        assert item.get_aging() == item.ingress_time + item.age

    def test_aging_is_measured_from_the_last_comm_tick_not_the_ingress_argument(self, link):
        """`ingress` stamps `self._clock`, which only `comm_tick` advances.

        The `sim_time` argument is used for the drop draw and the drop log, not
        for the arrival deadline. A caller that ingresses without ticking first
        gets a deadline measured from time zero.
        """
        link.ingress(sim_time=500, msg=_msg(1))
        assert link.memory[0][0].get_aging() == DELAY_MS  # not 500 + DELAY_MS

        link.comm_tick(1000)
        link.ingress(sim_time=7, msg=_msg(2))
        assert link.memory[0][-1].get_aging() == 1000 + DELAY_MS

    def test_ingress_reports_success_for_a_dropped_packet(self, sink):
        """A drop is a processed packet, not a rejected one -- the sender must
        not retry it as though the link had refused it."""
        link = Communication(0.05, "exp", _cfg(drop_rate=1.0))

        assert link.ingress(sim_time=10, msg=_msg(1)) is True

    def test_dropped_packet_is_recorded_and_not_queued(self, sink):
        link = Communication(0.05, "exp", _cfg(drop_rate=1.0))

        link.ingress(sim_time=10, msg=_msg(7))

        assert link.memory == {}
        assert link.dropped_packets == [{"sim_time": 10, "sender": "sensor", "id": 7}]

    def test_ingress_is_refused_while_offline(self, link):
        link.online = False

        assert link.ingress(sim_time=0, msg=_msg(1)) is False
        assert link.memory == {}

    def test_multiple_recipients_have_independent_queues(self, link):
        link.ingress(sim_time=0, msg=_msg(1), recipient=0)
        link.ingress(sim_time=0, msg=_msg(2), recipient=1)
        link.comm_tick(DELAY_MS)

        assert [m.id for m in link.get_messages(receiver=0)] == [1]
        assert [m.id for m in link.get_messages(receiver=1)] == [2]


class TestDeliveryTiming:

    def test_message_is_withheld_before_its_aging_time(self, link):
        link.ingress(sim_time=0, msg=_msg(1))

        link.comm_tick(DELAY_MS - 1)

        assert link.get_messages() == []
        assert len(link.memory[0]) == 1

    def test_message_is_delivered_on_the_tick_it_reaches_its_aging_time(self, link):
        """The comparison is `clock >= aging`, so the deadline tick delivers."""
        link.ingress(sim_time=0, msg=_msg(1))

        link.comm_tick(DELAY_MS)

        assert [m.id for m in link.get_messages()] == [1]

    def test_message_is_never_delivered_twice(self, link):
        for msg_id in range(4):
            link.ingress(sim_time=0, msg=_msg(msg_id))
        link.comm_tick(DELAY_MS)

        first = link.get_messages()
        second = link.get_messages()

        assert len(first) == 4
        assert second == []
        assert link.memory[0] == []

    def test_a_late_tick_still_delivers_everything_already_due(self, link):
        link.ingress(sim_time=0, msg=_msg(1))

        link.comm_tick(DELAY_MS * 50)

        assert [m.id for m in link.get_messages()] == [1]

    def test_unknown_receiver_yields_no_messages_without_raising(self, link):
        assert link.get_messages(receiver=99) == []

    def test_zero_delay_message_is_available_on_the_ingress_tick(self, sink):
        link = Communication(0.05, "exp", _cfg(delay=0))
        link.comm_tick(250)

        link.ingress(sim_time=250, msg=_msg(1))

        assert [m.id for m in link.get_messages()] == [1]


class TestPruningUnderSetSemantics:
    """The behaviour the harness's known-noise normalisation sits on top of."""

    def test_every_ready_message_is_delivered_exactly_once(self, link):
        expected = set(range(25))
        for msg_id in sorted(expected):
            link.ingress(sim_time=0, msg=_msg(msg_id))
        link.comm_tick(DELAY_MS)

        delivered = [m.id for m in link.get_messages()]

        assert sorted(delivered) == sorted(expected)
        assert len(delivered) == len(set(delivered)), "set pruning must not duplicate"

    def test_pruning_retains_every_message_that_is_not_yet_ready(self, link):
        for msg_id in range(5):  # due at DELAY_MS
            link.ingress(sim_time=0, msg=_msg(msg_id))
        link.comm_tick(50)
        for msg_id in range(5, 10):  # due at 50 + DELAY_MS
            link.ingress(sim_time=50, msg=_msg(msg_id))

        link.comm_tick(DELAY_MS)
        delivered = {m.id for m in link.get_messages()}

        assert delivered == {0, 1, 2, 3, 4}
        assert {item.payload.id for item in link.memory[0]} == {5, 6, 7, 8, 9}

    def test_nothing_is_lost_when_many_messages_share_one_deadline(self, link):
        """Set difference on identity-hashed items must drop exactly the ready
        ones, even when they are indistinguishable by arrival time."""
        for msg_id in range(40):
            link.ingress(sim_time=0, msg=_msg(msg_id))
        link.comm_tick(50)
        for msg_id in range(40, 60):
            link.ingress(sim_time=50, msg=_msg(msg_id))

        link.comm_tick(DELAY_MS)
        first_wave = {m.id for m in link.get_messages()}
        link.comm_tick(50 + DELAY_MS)
        second_wave = {m.id for m in link.get_messages()}

        assert first_wave == set(range(40))
        assert second_wave == set(range(40, 60))
        assert link.memory[0] == []

    def test_delivered_set_and_arrival_ticks_are_stable_across_identical_runs(self, sink):
        """Message identity and arrival tick are the invariant; list order is not.

        This is the property `compare_episodes.py` actually needs. If it ever
        fails, the ego message log has a real regression rather than the known
        ordering churn.
        """

        def run():
            link = Communication(0.05, "exp", _cfg())
            arrivals = set()
            pending = iter(range(12))
            for tick in range(0, 400, 25):
                link.comm_tick(tick)
                for message in link.get_messages():
                    arrivals.add((tick, message.id))
                nxt = next(pending, None)
                if nxt is not None:
                    link.ingress(sim_time=tick, msg=_msg(nxt))
            return arrivals

        assert run() == run()
        assert run(), "the schedule must actually deliver something"

    def test_delivery_order_is_not_asserted_but_content_is_recoverable(self, link):
        """Sorting by message id recovers a stable sequence from an unstable list.

        This mirrors what the harness does before diffing.
        """
        for msg_id in range(30):
            link.ingress(sim_time=0, msg=_msg(msg_id))
        link.comm_tick(DELAY_MS)

        delivered = link.get_messages()

        assert [m.id for m in sorted(delivered, key=lambda m: m.id)] == list(range(30))


class TestOfflineGating:

    def test_get_messages_withholds_rather_than_discards_while_offline(self, link):
        link.ingress(sim_time=0, msg=_msg(1))
        link.comm_tick(DELAY_MS)
        link.online = False

        assert link.get_messages() == []
        assert len(link.memory[0]) == 1, "the queue must survive the outage"

    def test_queued_messages_survive_an_offline_interval(self, link):
        link.ingress(sim_time=0, msg=_msg(1))
        link.online = False
        link.get_messages()

        link.online = True
        link.comm_tick(DELAY_MS)

        assert [m.id for m in link.get_messages()] == [1]


class TestEventParsing:

    def test_offline_event_takes_the_link_down(self, link):
        link.parse_event(ComEvent.OFFLINE)
        assert link.online is False

    def test_online_event_brings_the_link_back(self, link):
        link.online = False
        link.parse_event(ComEvent.ONLINE)
        assert link.online is True

    @pytest.mark.parametrize("initial", [True, False])
    def test_none_event_leaves_the_link_state_unchanged(self, link, initial):
        link.online = initial
        link.parse_event(ComEvent.NONE)
        assert link.online is initial

    @pytest.mark.parametrize("event", [ComEvent.OFFLINE, ComEvent.ONLINE])
    def test_repeating_an_event_is_idempotent(self, link, event):
        link.parse_event(event)
        state = link.online
        link.parse_event(event)
        assert link.online is state


class TestCommTick:

    def test_clock_advances_before_the_fault_model_is_stepped(self, link):
        """`comm_tick` sets `_clock` first, so a fault model that inspects the
        link sees the current tick, not the previous one."""
        stub = _StubFault([ComEvent.ONLINE], clock_probe=lambda: link._clock)
        link.fault_model = stub

        link.comm_tick(700)

        assert stub.seen_times == [700]
        assert stub.seen_clocks == [700]

    def test_tick_applies_the_fault_model_event(self, link):
        link.fault_model = _StubFault([ComEvent.OFFLINE, ComEvent.ONLINE])

        link.comm_tick(10)
        assert link.online is False

        link.comm_tick(20)
        assert link.online is True

    def test_each_tick_logs_exactly_one_entry(self, link, sink):
        for tick in (0, 50, 100):
            link.comm_tick(tick)

        assert [entry["t"] for entry in sink.entries] == [0, 50, 100]

    def test_log_reports_dropped_packets_then_clears_the_buffer(self, sink):
        link = Communication(0.05, "exp", _cfg(drop_rate=1.0))
        link.ingress(sim_time=5, msg=_msg(1))
        link.ingress(sim_time=6, msg=_msg(2))

        link.comm_tick(10)

        assert [d["id"] for d in sink.entries[-1]["dropped"]] == [1, 2]
        assert link.dropped_packets == []

    def test_drop_log_does_not_carry_over_to_the_next_tick(self, sink):
        link = Communication(0.05, "exp", _cfg(drop_rate=1.0))
        link.ingress(sim_time=5, msg=_msg(1))
        link.comm_tick(10)

        link.comm_tick(20)

        assert sink.entries[-1]["dropped"] == []

    def test_log_records_the_link_state(self, link, sink):
        link.fault_model = _StubFault([ComEvent.OFFLINE, ComEvent.ONLINE])

        link.comm_tick(10)
        link.comm_tick(20)

        assert [entry["online"] for entry in sink.entries] == [False, True]


class TestBiasConfiguration:

    def test_bias_without_a_sampler_is_a_configuration_error(self, sink):
        with pytest.raises(SimulationConfigurationError, match="Communication bias"):
            Communication(0.05, "exp", _cfg(), sampler=None, bias_cfg={"network_bias": {"delay": 5}})

    def test_empty_bias_block_is_treated_as_nominal(self, sink):
        link = Communication(0.05, "exp", _cfg(), sampler=None, bias_cfg={})

        assert link.network_model.sampler is None
        assert link.fault_model.sampler is None

    def test_network_and_fault_bias_blocks_are_routed_to_their_models(self, sink):
        class _Sampler:
            def normal(self, **kwargs):
                return 900.0

        link = Communication(
            0.05,
            "exp",
            {"network": {"type": "uniform", "packet_drop_rate": 0.1, "delay": 10, "jitter": 4},
             "fault": {"type": "sit", "offline": 50, "trigger_mean": 1000, "trigger_var": 100}},
            sampler=_Sampler(),
            bias_cfg={
                "network_bias": {"packet_drop_rate": 0.8},
                "fault_bias": {"trigger_mean": 200},
            },
        )

        assert link.network_model.drop_rate_bias == 0.8
        assert link.network_model.drop_rate == 0.1
        assert link.fault_model.bias_cfg == {"trigger_mean": 200}
        assert link.fault_model.trigger_mean == 1000, "the base law must be untouched"

    def test_a_missing_bias_section_leaves_that_model_nominal(self, sink):
        class _Sampler:
            def normal(self, **kwargs):
                return 900.0

        link = Communication(
            0.05,
            "exp",
            {"network": {"type": "uniform", "packet_drop_rate": 0.1, "delay": 10, "jitter": 4},
             "fault": {"type": "sit", "offline": 50, "trigger_mean": 1000, "trigger_var": 100}},
            sampler=_Sampler(),
            bias_cfg={"fault_bias": {"trigger_mean": 200}},
        )

        assert link.network_model.drop_rate_bias == link.network_model.drop_rate
