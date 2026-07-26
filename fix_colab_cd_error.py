"""
Colabでの%cdコマンドの誤りを修正するためのヘルパースクリプト
"""

def fix_cd_command():
    """正しいディレクトリに移動するための修正関数"""
    import os
    
    # 現在のディレクトリを確認
    current_dir = os.getcwd()
    print(f"現在のディレクトリ: {current_dir}")
    
    # /contentcontent/slm のような重複パスがあれば修正
    if '/contentcontent/' in current_dir:
        correct_dir = current_dir.replace('/contentcontent/', '/content/')
        try:
            os.chdir(correct_dir)
            print(f"ディレクトリを修正: {os.getcwd()}")
        except:
            print(f"ディレクトリ {correct_dir} は存在しません")
    
    # slmディレクトリが存在すればそこに移動
    elif os.path.exists('/content/slm'):
        os.chdir('/content/slm')
        print(f"slmディレクトリに移動: {os.getcwd()}")
    
    # インストール確認
    try:
        import slm
        print(f"slmパッケージがインストールされています（バージョン: {slm.__version__}）")
    except:
        print("slmパッケージがインストールされていません")
        print("以下のコマンドを実行してインストールしてください:")
        print("!pip install -e .")

if __name__ == "__main__":
    fix_cd_command()
