import os
from pathlib import Path

def clean_dataset_fast(main_folder, dry_run=True):
    """
    使用集合运算快速删除无对应mask的图片（不遍历比对）
    """
    main_path = Path(main_folder)
    subsets = ['train', 'test', 'query']
    
    print(f"{'='*60}")
    print(f"{'[预览模式]' if dry_run else '[删除模式]'} 快速清理数据集: {main_path}")
    print(f"{'='*60}\n")
    
    for subset in subsets:
        img_folder = main_path / subset
        mask_folder = main_path / 'masks' / 'pifpaf_maskrcnn_filtering' / subset
        
        if not img_folder.exists() or not mask_folder.exists():
            print(f"⚠️  跳过 {subset}: 文件夹不存在")
            continue
        
        # 1. 直接获取文件名集合（不含扩展名）
        img_names = {p.stem for p in img_folder.glob('*.jpg')}
        mask_names = {p.stem for p in mask_folder.glob('*.npy')}
        
        # 2. 集合运算找出需要删除的文件
        to_delete = img_names - mask_names
        
        if not to_delete:
            print(f"✅ {subset}: 所有 {len(img_names)} 张图片都有对应mask")
            continue
        
        # 3. 批量删除
        print(f"🗑️  {subset}: 发现 {len(to_delete)} 张图片需要删除")
        print(to_delete)
        
        for name in to_delete:
            img_path = img_folder / f"{name}.jpg"
            print(f"  ❌ 删除: {img_path.name}")
            if not dry_run:
                img_path.unlink(missing_ok=True)
        
        print(f"✅ {subset}: 保留 {len(img_names) - len(to_delete)} 张, 删除 {len(to_delete)} 张\n")
    
    print(f"{'='*60}")
    print("处理完成！")
    if dry_run:
        print("⚠️  当前为预览模式，未实际删除任何文件！")
        print("确认无误后，设置 dry_run=False 执行实际删除")
    print(f"{'='*60}")

if __name__ == "__main__":
    dataset_root = "/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/datasets/DaJixiang"
    # clean_dataset_fast(dataset_root, dry_run=True)
    clean_dataset_fast(dataset_root, dry_run=False)  # 确认后执行