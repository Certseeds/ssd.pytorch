function test_small()
{
    python3 .\test.py --dataset barcode `
    --img_list_file_path data\barcode\CorrectDetect.txt `
    --trained_model weights\ssd300_barcode\349.pth `
    --save_folder test `
    --cuda true `
    --test_or_eval false # if want to work with ap_*, make it `false`
}
function test_mmu()
{
    python3 .\test.py --dataset barcode `
    --img_list_file_path data\barcode\valid.txt `
    --trained_model weights\ssd300_barcode\348.pth `
    --save_folder test `
    --cuda true `
    --test_or_eval false # if want to work with ap_*, make it `false`
}
function test_all()
{
    python3 .\test.py --dataset barcode `
    --trained_model weights\ssd300_barcode\349.pth `
    --img_list_file_path data\barcode\train.txt `
    --save_folder test `
    --cuda true
}
function ap_small()
{
    python3 .\ap_test.py `
    --img_list_file_path data\barcode\CorrectDetect.txt `
    --pred_label_path test\barcode11\labels
}
function ap_MMU()
{
    python3 .\ap_test.py `
    --img_list_file_path data\barcode\valid.txt `
    --pred_label_path test\barcode14\labels
}
function small()
{
    test_small
    ap_small
}
function MMU()
{
    test_mmu
    ap_MMU
}
function train_barcode()
{
    python3 ./train.py --dataset barcode `
    --batch_size 16 `
    --num_workers 4 `
    --cuda true `
    --lr 1e-5 `
    --img_list_file_path "./data/barcode/train_with_wwu.txt"
}
Write-Output("114514")
MMU
# train_barcode
# ./win.ps1 | Out-File -FilePath ./log.log