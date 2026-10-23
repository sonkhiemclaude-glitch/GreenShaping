import pytest
from greenshaping.data.capture_windows import account, compare, validate_windows


def packet(t, size=100, fingerprint='a'):
    return dict(epoch=t, network_bytes=size, direction='up', fingerprint=fingerprint)


def test_windows_keep_cover_and_gaps_separate_and_do_not_double_count_boundary():
    w=[dict(id='a',start_epoch=1,end_epoch=2),dict(id='b',start_epoch=2,end_epoch=3)]
    x=account([packet(0),packet(1),packet(2),packet(3)],w)
    assert [r['packets'] for r in x['per_session']]==[1,1]
    assert x['outside_session_bytes']==200 and x['total_bytes']==400
    assert not x['lifecycle_coverage_verified']


def test_missing_lifecycle_and_packet_loss_fail_even_when_byte_totals_match():
    w=[dict(id='a',start_epoch=1,end_epoch=3)]
    life=dict(capture_ready_epoch=0,capture_stopped_epoch=4,kernel_dropped_packets=0)
    a=account([packet(2,fingerprint='a')],w,life)
    b=account([packet(2,fingerprint='b')],w,life)
    c=compare(a,b)
    assert c['byte_difference']==0 and c['left_only_packets']==1 and not c['strict_comparison_passed']
    assert compare(a,a)['strict_comparison_passed']
    b=account([packet(2)],w)
    assert not compare(a,b)['strict_comparison_passed']
    life['kernel_dropped_packets']=1
    assert not account([packet(2)],w,life)['lifecycle_coverage_verified']


@pytest.mark.parametrize('w',[
 [],[dict(id='x',start_epoch=1,end_epoch=1)],
 [dict(id='x',start_epoch=1,end_epoch=3),dict(id='y',start_epoch=2,end_epoch=4)],
 [dict(id='x',start_epoch=1,end_epoch=2),dict(id='x',start_epoch=3,end_epoch=4)],
 [dict(id='x',start_epoch=float('nan'),end_epoch=2)]])
def test_invalid_windows_are_not_silently_accepted(w):
    with pytest.raises(ValueError):validate_windows(w)
