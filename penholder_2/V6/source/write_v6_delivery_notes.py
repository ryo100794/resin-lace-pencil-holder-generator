#!/usr/bin/env python3
"""Write version-specific Japanese notes from verified V6 output reports."""
from pathlib import Path
import collections,json,hashlib

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'

def report(name,passed=True):
    d=json.loads((ROOT/'reports'/name).read_text())
    if passed and d.get('passed') is not True:raise ValueError('Required gate not passed: '+name)
    return d

def point(p):return '('+', '.join(f'{float(v):.3f}' for v in p)+')'

def model_point(p):
    """Retain the saved model coordinates in the detailed band-cut record."""
    return '('+', '.join(repr(float(v)) for v in p)+')'

def endpoint_band_entry(band):
    """Expose exact model values without suggesting achievable hand accuracy."""
    return dict(endpoint_index=band['endpoint_index'],
        center_mm=list(band['center_mm']),plane_normal=list(band['axis_toward_free_body']),
        normal_direction='toward_free_body',modeled_removed_band_width_mm=band['kerf_width_mm'],
        waist_length_mm=band['waist_length_mm'],waist_radius_mm=band['waist_radius_mm'],
        minimum_margin_to_waist_end_mm=band['minimum_axial_margin_to_waist_end_mm'],
        fixed_side_plane_offset_mm=band['fixed_side_plane_offset_mm'],
        free_side_plane_offset_mm=band['free_side_plane_offset_mm'],
        modeled_cutter_radial_allowance_mm=band['cutter_radial_allowance_mm'],
        width_basis=band['width_basis'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assembly=report('custom_support_assembly.json')
    topology=report('custom_supports_validation.json')
    export=report('v6_export_validation.json')
    external=report('v6_external_removal_validation.json')
    service=report('v6_internal_service_validation.json')
    precision=report('v6_precision_motion_validation.json')
    preparation=report('v6_precision_preparation.json')
    report('v6_release_validation.json')
    report('v5_mechanism_unchanged.json')
    holder=report('custom_holder_supports.json',passed=False)
    if holder.get('generation_checks_passed',holder.get('passed')) is not True:
        raise ValueError('Holder local generation is not valid')
    spheres=[report(f'custom_sphere_supports_{n}.json') for n in (3,4,5)]
    hc=collections.Counter(x['kind'] for x in holder['parts'])
    sphere_temporary=3*sum(s['removable_support_count'] for s in spheres)
    sphere_micro=3*sum(s['fixed_micro_reinforcement_count'] for s in spheres)
    holder_micro=hc['fixed_micro_reinforcement']
    base_tacks=hc['plate_perimeter_tack']
    holder_twigs=sum(v for k,v in hc.items() if k not in ['fixed_micro_reinforcement','plate_perimeter_tack'])
    new_neck_values=[]
    for r in holder['parts']:
        if r.get('minimum_neck_minor_width_lower_bound_mm') is not None:
            new_neck_values.append(float(r['minimum_neck_minor_width_lower_bound_mm']))
    for s in spheres:
        for r in s['supports']:
            if r.get('minimum_neck_minor_diameter_normal_to_axis_mm') is not None:
                new_neck_values.append(float(r['minimum_neck_minor_diameter_normal_to_axis_mm']))
    waist=f'{min(new_neck_values):.3f}' if new_neck_values else '報告書を参照'
    dims=' × '.join(f'{v:.3f}' for v in export['dimensions_mm'])
    segmented=sum(bool(p['additional_body_cuts']) for p in external['plans'])
    operation_count=len(external['plans'])
    if operation_count!=external['planned_support_count']:
        raise ValueError('Removal plan count does not match its certified rows')
    cut_only=[p for p in external['plans'] if p['mode']=='endpoint_cuts_only']
    base_actual_stock=[p for p in external['plans'] if p.get('finishing_stock_partition')]
    band_operations=[]
    for p in external['plans']:
        bands=p.get('endpoint_cut_bands',[])
        if p['mode']=='endpoint_cuts_only' and (not bands or p['fragments'] or p['additional_body_cuts']):
            raise ValueError('Cut-only operation has inconsistent cut/extraction data: '+p['id'])
        if not bands:continue
        if (p.get('endpoint_cut_validation') or {}).get('passed') is not True:
            raise ValueError('Finite endpoint-cut validation is missing: '+p['id'])
        band_operations.append(dict(id=p['id'],sequence_index=p['sequence_index'],
            mode=p['mode'],endpoint_bands=[endpoint_band_entry(band) for band in bands],
            requires_free_body_extraction=bool(p['fragments']),free_fragment_count=len(p['fragments']),
            additional_body_cut_count=len(p['additional_body_cuts']),
            after_endpoint_cuts_ja=('記載したくびれを局所的に切断したら、この番号の作業は終了。モデル上、別に移動する取り出し片はなく、固定側に残る接点は引き抜かない。'
                if p['mode']=='endpoint_cuts_only' else
                ('端部の処理後、指定箇所で胴を分割し、各片を順に移動する。' if p['additional_body_cuts'] else
                 '端部の処理後、取り出し片を指定方向へ順に移動する。'))))
    band_widths=[b['modeled_removed_band_width_mm'] for p in band_operations for b in p['endpoint_bands']]
    holder_band_count=sum(p['group']=='holder' and bool(p.get('endpoint_cut_bands')) for p in external['plans'])
    band_range=[min(band_widths),max(band_widths)] if band_widths else None
    band_range_text=(f'{band_range[0]:.3f}〜{band_range[1]:.3f} mm' if band_range else '該当なし')
    base_extracted=sum(p['group']=='holder' and p['kind']=='plate_perimeter_tack' for p in external['plans'])
    base_flush_finishing=base_tacks-base_extracted
    original=report('support_analysis_v5_baseline.json',passed=False)
    negative=topology.get('negative_cavity_birth_count','報告書参照')
    residual=topology.get('positive_non_plate_material_birth_count','報告書参照')
    module_lines=[]
    for s in spheres:
        n=s['module_layers'];d={3:18,4:22,5:26}[n]
        module_lines.append(f'・{n}層・基準径{d} mm、3個：外部仮支持{s["removable_support_count"]}本/球、残す小補強{s["fixed_micro_reinforcement_count"]}個/球。')
    note=f'''象牙風多層球ペン立て V6 — モデル専用サポート
単位 mm / 造形方向はモデル座標 +Z / 底面を先に造形

使用するデータ
  ivory_penholder_v6_custom_supported.3mf
  ivory_penholder_v6_custom_supported.stl（通常のZIPを展開して使用）
3MFを優先してください。STLは単位mm・倍率100%で読み込みます。
支持はモデルに組み込んであります。この向きではスライサーの自動サポート生成をOFFにし、
配置済み支持を含むモデル全体をそのままスライスすることを想定しています。
露光・積層ピッチ・リフト/剥離・洗浄・二次硬化は機種と樹脂に合わせて設定してください。
機種、樹脂、印刷された過去版、破損工程は確認できていません。実機での印刷成功を保証する版ではありません。

1. 今回の変更

V5の保存済み造形メッシュを固定した原本とし、固定外殻と本体の細工に支持を追加しました。
内側の回転層、対向する2点の軸受、内部の既存仮支持81本・接点162か所の設計元データは変更していません。
球殻の基本厚1.20 mm、3/4/5層の構成、9球の配置を維持しています。
基本厚は軸穴、作業窓、装飾、局所修正を含む全箇所の最小肉厚を意味しません。
出力メッシュには各回{preparation['simplification_requested_tolerance_mm']:g} mmを指定した微小な面整理を{preparation.get('simplification_pass_count',1)}回と、float32丸めを行っています。
面整理の各回の誤差上限は合計して評価しています。
元の可動層キャッシュから今回の出力までの面位置誤差上限は{precision['per_surface_error_upper_bound_mm']:.9f} mmです。
この表現上の変化を含めて、回転隙間を再評価しています。

・球の固定外殻：今回の外部仮支持は計{sphere_temporary}本。
・本体の装飾・長い放射部材：今回の仮支持は計{holder_twigs}本。
・底面の細かい先端：プレートへ届く短い接地部を{base_tacks}か所追加。
・固定装飾の小さな隙間：残す小補強は本体{holder_micro}個、球外殻{sphere_micro}個。
・回転層どうしを恒久的につなぐ小補強は追加していません。
{chr(10).join(module_lines)}

長い柱を全面に立てる方式ではなく、原則として近くの先に造形される固定部分から短く支えます。
元からある上縁の横糸を使える箇所では、その接続を確認して重複する支持を省いています。
今回追加した仮支持のくびれは、傾きと多角形断面を考慮した最細幅の下限が{waist} mmです。
この寸法は採用形状の幾何学値であり、あらゆる樹脂・装置で十分な強度となる下限ではありません。
くびれ前後の断面変化は接点近くへ切断/曲げを集中させるための形状です。実際の破断位置・荷重は未測定です。

2. 画像の見方

象牙色：元のV5本体・球・内部支持。
オレンジ：今回追加した仮支持、または底面を整えるときに処理する短い接地部。
青：固定装飾に残す小補強。切り取りません。
これらは説明用の色です。印刷用3MF/STLは一体の支持付きメッシュで、多材料指定ではありません。
実メッシュから全体、5層球、壁、底面の画像を作成しています。

3. 造形方向と底面

読み込んだ向きのまま、モデル底面Z=0を造形面へ置きます。
モデル内で底面→胴→上縁の順に造形します。装置で天地逆に吊り下がって見えても、この積層順です。
球の局所座標では+Yが造形方向で、配置後は本体+Zに一致します。
傾ける、反転する、縮小する、モデルをラフトから浮かせる場合は今回の接続検査の前提が変わります。
底面は多数の接点を持つ元の細工です。疎らな数本の脚で全体を持ち上げる構成にはしていません。
プレート接着、底面露光、剥離の条件は使用装置で確認してください。
全体寸法：{dims} mm（X × Y × Z、装飾と今回の支持を含む）。

4. 外部の支持を取り外す

造形プレートから外した後のモデルを対象にした手順です。
REMOVE_ORDER_JA.txtとsupport_removal_index.jsonに、実形状から検査した順序、接点、分割位置、移動方向を記録しています。
Hで始まる番号は本体、S3/S4/S5はそれぞれ3/4/5層球の外部支持です。
Aに続く数字は球の本体まわりの配置角度です。全ての座標は印刷モデルと同じmmです。
テキスト手順の座標・方向・距離には読みやすさのため丸めた参照値を含みます。検査に使った数値はsupport_removal_index.jsonを参照してください。

支持側を保持し、記載された細いくびれを局所的に切断してから、取り出し片があるものを取り出します。
球殻を握ってねじる、未切断の支持をまとめて引く、軸受や作業窓の補強帯を切る操作は避けます。
取り外し手順は計{operation_count}件で、このうち{len(cut_only)}件は「端部の切断のみ」です。
切断のみの番号は、くびれの処理で終了します。モデル上、別に取り出して移動する片はありません。
固定側に残る接点を引き抜かないでください。

球の外部支持と本体の{holder_band_count}本では、首元の切断を「有限幅の帯を除去する」として表しています。
帯の幅、中心、法線などは詳細手順とJSONにある検査モデルの参照値で、その幅を手作業で加工する指示ではありません。
実際の工具の入り方、切断力、樹脂の欠けやすさは未検証です。数値の桁数も実作業で達成できる精度を意味しません。

外部支持のうち{segmented}本は、記載した端部を処理した後に支持の胴を追加で切る手順です。
複数片に分けるものと、端などを切り詰めて一つの取り出し片を残すものがあります。
分割切断は支持の胴に指定しており、元の装飾を切る指示ではありません。
複数片になる支持は、一覧の「片1→片2…」の順で外します。先に外した片の空間を、後続の片の取り出しに使う場合があります。
各片の最初の中心座標と範囲も一覧に記載しています。
底面の短い接地部は下面から仕上げます。高さが極小のものは全面をむしり取らず、下面を整える対象です。
底面の{len(base_actual_stock)}か所は、元の本体を含めない「実際の追加材」を仕上げ対象として区別しています。
本体と重なる部分は残します。この区別はモデル上の仕上げ範囲で、複雑な接合面に沿う実工具の入り方や切断幅の検証ではありません。
底面の接地部{base_tacks}か所のうち{base_flush_finishing}か所は、独立した取り出し片を持たない薄い仕上げ接点です。
青い固定小補強は残します。

取り出し検査は、指定順に外した後の残り、未除去の全支持、本体、球の実形状を障害物として扱っています。
ペン収納部へ取り込む経路では、その後に上の開口から出す移動も確認しています。
切断面、端部の帯幅、胴の分割幅はモデル上の値です。出力誤差や実機の寸法誤差、実際の刃幅への適合は未検証です。
工具本体・柄・指の大きさ、刃の入り方、実際の破断力はこの検査に含みません。
くびれが工具で扱えない、樹脂が欠ける、抵抗がある場合は殻を力で回して解放しないでください。

5. 内部支持を取り外して回転を解放する

今回の外部仮支持を先に外します。その後、V5と同じ内部支持の手順に進みます。
最も内側の層間から外側へ、各層間の前面1本→底部左→底部右の順です。
1本ずつ中央を保持し、両端の首を切り離してから球の正面へ取り出します。
切断・取り出し経路は、各層が配布モデルの初期姿勢にある状態で検査しています。
作業中はその向きを保ち、全ての内部支持を外した後に1層ずつ回転を確認します。
前面の内部支持：胴径0.55 mm・首径0.40 mm。
底部の内部支持：胴0.82 × 0.76 mm・首0.74 × 0.54 mm。
内部支持の切断残りは公称0.15 mm、V5の回転検査では0.20 mmまでの残りを含みます。
本体の固定枠、作業窓の補強、底部の受け座、軸、軸受は残します。
元の工具経路は先端径0.20 mm・軸径0.50 mmを基準にした幾何学検査です。
市販の全てのニッパーやペンチの作業空間を意味しません。

6. 検証したことと、未確認のこと

・V5の球の部品キャッシュを全件ハッシュ照合。機構の元データを維持。
・元メッシュに宣言した追加形状だけを合成。その後の微小な面整理・丸めは別に誤差上限を記録。
・造形用float32精度で、閉じた面、面の向き、接続、退化を確認。
・全体の面の接続変化を追跡し、空洞の出現と材料の出現を区別。
  比較元V5のプレート以外から始まる材料出現：{original['positive_non_plate_material_birth_count']}件。
  V6で同じ検査に残った材料出現：{residual}件。空洞の出現は{negative}件で、材料の孤立とは区別。
  体積閾値1e-12 mm³、接地判定高さ1e-6 mm、面の接続変化の高さ分解能1e-8 mmでの幾何学検査です。
  数値閾値はプリンターの再現精度ではありません。
・実際の軸受の遊び・傾きと回転を含めた可動範囲に、新しい支持が侵入しないことを検査。
  元版の丸め、今回の面整理、今回の丸めを累積し、最終メッシュのハッシュに結びつけて隙間を再評価。
・今回の外部支持の取り出し順序と連続移動を検査。
・底面の実際の追加材を元本体から区別し、本体と残す材料の全体形状が変わらないことを検査。
  凹形状の取り出しは、開始形状と各境界面の移動を覆う小片を使い、各小片の干渉量の合計で判定。
  数値計算のために小片を広げる場合も、広げた分を干渉量に含めています。印刷形状を広げる処理ではありません。
・球の外部支持と本体の{holder_band_count}本は、有限幅の端部切断帯・残す接点・取り出し片に分け、材料の欠落や重複と残りの範囲を検査。
  本体のこれらの支持は印刷形状を変えずに切断後の分割を定義し、残る接点が従来の最大残存範囲内に収まることを確認。
・外部支持を切った後の最大残りと固定小補強が、元の内部切断/取り出し経路を塞がないことを検査。
・出力した3MF/STLを読み直し、全三角形の座標・向きを照合。

孤立形状が無いことは、剥離荷重に耐えることの証明ではありません。
樹脂の強度、露光による隙間の閉塞、摩擦、収縮、剥離、実際の切断力、印刷成功率は未検証です。
有限要素解析や実機試験の結果を添付したものではありません。
今回の支え方が以前の破損原因を特定したという意味でもありません。

7. ファイルと再生成

Google Drive保存先：workspace/penholder_2/V6
V2〜V5はそれぞれの版フォルダーを維持。V6/work_historyは作業途中の記録で、印刷には使いません。
印刷用3MF、STLのZIP、再編集用の通常ZIP 2本、説明書、実形状プレビューを分けて保存しています。
印刷だけなら3MF、またはSTLのZIPを使います。再編集用ZIPは印刷に必須ではありません。
再編集・再生成には、次の両方を同じ親フォルダーへ展開し、projectフォルダーの内容を合わせます。
  ivory_v6_project.zip：原本の固定メッシュ、球部品、ソース、検査報告、説明書。
  ivory_v6_support_assets.zip：採用した支持部品、切断片、取り出し経路の元データ。
分割バイナリではなく、どちらも単独で開ける通常のZIPです。再生成には両方が必要です。
FILE_MANIFEST.jsonに必須の支持データZIPのハッシュを記録し、ASSET_MANIFEST.jsonにその中身を記録しています。
失敗した試作支持や以前のファン状支持を印刷用の入力には含めません。

上記2本のZIPを合わせて展開したprojectフォルダーで：
  python -m pip install -r source/requirements.txt
  python source/rebuild_v6.py
画像も作る場合：
  python source/rebuild_v6.py --render
保持済みの採用支持から組立・検査・出力を再現します。支持探索の再実行は別オプションです。
同梱の取り外し指示を探索の初期候補として使い、切断・移動・退出は再構築時に全件再検証します。
接合部修正と固定接点の小補強の旧入力・採用片は、別添のivory_v6_repair_history.zipに保存しています。
履歴ZIPは別のフォルダーに展開してください。現在のprojectへ重ねると、同じ名前の報告書を上書きする場合があります。
履歴の修正スクリプトを当時の入力から再実行する場合は、
配布対象外の当時の全体診断メッシュを別途再生成する必要があります。通常のV6再構築には不要です。
全高接続検査にはC++17対応のg++が必要です。Linuxのヘッドレス描画にはXvfbなどが必要です。
検査報告はreports、個々の外部支持と取出し片はcache内です。
元のV5機構の報告とV6追加形状の報告は対象が異なります。各報告の入力SHA256とscopeを参照してください。
SHA256SUMS.txtで配布ファイルのハッシュを確認できます。

参考：支持位置・接触径・除去性の考え方
  https://formlabs.com/support/How-supports-work-in-SLA-printing/
  https://formlabs.com/support/Advanced-Support-Structure-Settings/
これらを使用装置の設定値として転用していません。
'''
    (OUT/'README_V6_JA.txt').write_text(note,encoding='utf-8')
    removal=['V6 外部支持の取り外し順序 / mm / モデル座標',
        '造形プレートから外した後のモデルを対象にした手順です。',
        '全て外した後に、最内層からV5内部支持を解放します。青い小補強は残します。',
        f'計{operation_count}件。端部の帯状切断があるものは{len(band_operations)}件、切断のみで終了するものは{len(cut_only)}件です。',
        '支持側を保持し、記載された細いくびれを局所的に切断します。球殻をねじって切り離さないでください。',
        f'端部切断帯の幅{band_range_text}は幾何学検査の参照値です。その幅を手作業で加工する指示ではありません。',
        '帯の中心・法線は保存済みモデルの数値です。表示桁数は実作業の精度を示しません。幅は法線方向の2平面間の距離です。',
        '実工具の刃幅・入り方・切断力は未検証です。実際の切断残りがモデルと異なる場合の取り出しやすさまでは確認していません。',
        '「端部の切断のみ」の番号には移動操作がありません。それ以外は必要な胴の追加切断後、各片を指定順に移動します。',
        '片の番号は取り外し順です。先に外した片の空間を次の片に使う場合があるため、この順序で扱います。各片の中心と範囲は切断直後のモデル座標です。',
        '元の移動ベクトルは単位方向で、変位はその方向×距離です。このテキストの座標・方向・距離には丸めた参照値を含み、表示値から同じ経路を厳密に再現するものではありません。',
        '検査に使った丸め前のモデル数値はsupport_removal_index.jsonを参照してください。',
        '刃・柄・指の実際の作業空間や力は未検証です。支持以外の元の形は切りません。','']
    for p in external['plans']:
        removal.append(f'{p["sequence_index"]+1:04d}. {p["id"]} / {p["kind"]}')
        removal.append(f'  下側の接点 {point(p["root_mm"])} / 支える点 {point(p["target_mm"])}')
        if p.get('finishing_stock_partition'):
            removal.append('  下面仕上げ：元の本体と重なる材料を残し、実際に追加された支持部分だけを仕上げます。元本体を平らに削り取る指示ではありません。')
            removal.append('  仕上げ範囲と下方向の取り出しはモデルで検査。実工具の入り方、接合面に沿う刃幅、加工力は未確認です。')
        bands=p.get('endpoint_cut_bands',[])
        if bands:
            for k,band in enumerate(bands):
                removal.append(f'  端部切断帯{k+1}（モデル参照値）：幅 {band["kerf_width_mm"]:.3f} mm（法線方向） / 中心 {model_point(band["center_mm"])}')
                removal.append(f'    法線 {model_point(band["axis_toward_free_body"])}（支持の胴側向き）')
                removal.append(f'    くびれ長さ {band["waist_length_mm"]:.3f} mm / くびれ端までのモデル上の最小余裕 {band["minimum_axial_margin_to_waist_end_mm"]:.6f} mm')
        else:
            for k,c in enumerate(p['endpoint_cut_centers_mm']):
                normal=p.get('endpoint_cut_plane_normals',[])
                suffix=(' / 切断面の法線 '+point(normal[k])) if k<len(normal) else ''
                removal.append(f'  接点くびれ{k+1} {point(c)}'+suffix)
        if p['mode']=='endpoint_cuts_only':
            removal.append('  操作：端部の切断のみ。記載くびれの局所切断で終了し、別の片を取り出して移動する操作はありません。固定側に残る接点は引き抜きません。')
        else:
            if p['additional_body_cuts']:
                removal.append('  端部の処理後、次の胴の追加切断を行ってから各片を順に取り出します。')
            for k,c in enumerate(p['additional_body_cuts']):
                removal.append(f'  胴の追加切断{k+1} {point(c["centerline_reference_mm"])} / 分割面の法線 {point(c["plane_normal"])} / 除去帯幅 {c["modeled_removed_kerf_width_mm"]:.3f} mm')
            for k,f in enumerate(p['fragments']):
                bounds=f['bounds_mm']
                center=[(bounds[i]+bounds[i+3])/2 for i in range(3)]
                removal.append(f'  片{k+1}（取り外し順）の最初の位置：中心 {point(center)} / 範囲 {point(bounds[:3])} 〜 {point(bounds[3:])}')
                for j,l in enumerate(f['translation_legs']):
                    removal.append(f'  片{k+1}・移動{j+1}：方向 {point(l["direction"])} に {l["distance_mm"]:.3f} mm')
        removal.append('')
    (OUT/'REMOVE_ORDER_JA.txt').write_text('\n'.join(removal),encoding='utf-8')
    index=dict(version=6,units='mm',external_plan=external,
        fixed_micro_instruction='Blue fixed micro-reinforcements are permanent. Do not cut.',
        finite_endpoint_cutting=dict(operation_count=len(band_operations),band_count=len(band_widths),
            modeled_band_width_range_mm=band_range,cut_only_operation_count=len(cut_only),
            cut_only_ids=[p['id'] for p in cut_only],
            support_hold_instruction_ja='支持側を保持して細いくびれを局所的に切断し、球殻をねじって切り離さない。',
            model_condition_ja='指定した有限幅のくびれ帯を除去した状態を、接点の残りと取り出し経路の検査に用いる。',
            nominal_band_dimensions_are_analysis_references=True,
            physical_kerf_feasibility_certified=False,hand_cut_accuracy_certified=False,
            physical_scope_ja='手作業で指定幅を加工する要求ではない。実工具の刃幅・入り方・切断力は未検証で、数値の桁数は作業精度を意味しない。切断残りがモデル条件と異なる場合の取り出しやすさは未確認。',
            operations=band_operations),
        holder_source='cache/custom_holder_supports/holder_supports.json',
        sphere_sources=[f'cache/custom_sphere_supports/sphere_{n}layers_supports.json' for n in (3,4,5)])
    (OUT/'support_removal_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    summary=dict(passed=True,dimensions_mm=export['dimensions_mm'],holder_temporary_branches=holder_twigs,
        sphere_temporary_branches=sphere_temporary,base_start_tacks=base_tacks,
        permanent_fixed_micro_reinforcements=holder_micro+sphere_micro,
        minimum_new_temporary_neck_minor_width_mm=min(new_neck_values) if new_neck_values else None,
        original_internal_supports=81,external_removal_planned_count=external['planned_support_count'],
        endpoint_band_cut_operation_count=len(band_operations),endpoint_band_count=len(band_widths),
        holder_finite_endpoint_operation_count=holder_band_count,
        endpoint_band_modeled_width_range_mm=band_range,endpoint_cuts_only_operation_count=len(cut_only),
        external_operations_with_free_body_extraction=operation_count-len(cut_only),
        endpoint_cuts_only_ids=[p['id'] for p in cut_only],physical_kerf_feasibility_certified=False,
        base_contacts_without_separate_extracted_piece=base_flush_finishing,
        base_finishing_actual_added_stock_partitions=len(base_actual_stock),
        temporary_branches_requiring_additional_body_cuts=segmented,
        positive_non_plate_material_birth_count=residual,physical_print_tested=False)
    summary['complete_source_to_export_surface_error_upper_bound_mm']=precision['per_surface_error_upper_bound_mm']
    (ROOT/'reports/v6_delivery_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
