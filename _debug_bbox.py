"""诊断 USD 场景中 Base/Wall/Partition prim 的 BBox"""
from pxr import Usd, UsdGeom

usd_path = 'source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/scenes/real/shenxinfu/841837/start_open_seat.usda'
stage = Usd.Stage.Open(usd_path)

base_path = '/World/envs/env_0/room/Meshes/Base'
base_prim = stage.GetPrimAtPath(base_path)
if base_prim.IsValid():
    print(f'{base_path} VALID, children:')
    for child in base_prim.GetChildren():
        name = child.GetName()
        ptype = child.GetTypeName()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True, ignoreVisibility=True)
        bbox = cache.ComputeWorldBound(child)
        rng = bbox.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        dx, dy, dz = float(mx[0]-mn[0]), float(mx[1]-mn[1]), float(mx[2]-mn[2])
        tag = '<- WALL?' if ('Wall' in name or 'Partition' in name) else ''
        print(f'  {name}: type={ptype} dims=({dx:.3f},{dy:.3f},{dz:.3f}) mn=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) {tag}')
else:
    print(f'{base_path} INVALID')
    print('Searching for Wall/Partition prims...')
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if any(k in p for k in ['Wall', 'Partition']):
            print(f'  {p} type={prim.GetTypeName()}')
