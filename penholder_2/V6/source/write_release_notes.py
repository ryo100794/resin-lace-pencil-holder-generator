#!/usr/bin/env python3
"""Write Japanese release and trial-print notes from the final audited V5 files."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
STEM = 'ivory_multisphere_pen_holder_v5_printpath_20261025'


def read(name):
    return json.loads((ROOT/'reports'/name).read_text())


def main():
    gates = ['motion_validation.json', 'early_support_validation.json',
             'geometry_validation.json', 'holder_motion_clearance.json',
             'holder_service_access.json', 'precision_repairs_scope.json',
             'strength_validation.json', 'test_coupon_validation.json',
             'full_height_validation.json']
    for name in gates:
        if read(name).get('passed') is not True:
            raise RuntimeError('Complete geometric validation required: '+name)
    meta = [json.loads((ROOT/'cache'/'spheres'/f'sphere_{n}layers.json').read_text())
            for n in (3, 4, 5)]
    geom = read('geometry_validation.json')
    motion = read('motion_validation.json')
    early = read('early_support_validation.json')
    full = read('full_height_validation.json')
    baseline_full = read('full_height_v4_baseline.json')
    old_front_lead = max(
        row['material_precedes_first_permanent_contact_mm']
        for module in baseline_full['modules'] for row in module['wire_onset_diagnostics']
        if row['kind'] == 'front_window')
    wall = read('holder_motion_clearance.json')
    coupon = read('test_coupon_validation.json')
    dims = ' × '.join(f'{v:.3f}' for v in geom['dimensions_mm'])
    coupon_dims = ' × '.join(f'{v:.3f}' for v in coupon['export_mesh_validation']['dimensions_mm'])
    bearings = [v for m in motion['modules'] for v in m['measured_bearings']]
    rotations = [v for m in motion['modules'] for v in m['continuous_rotation']]
    radial_gap = min(v['all_angle_radial_clearance_bound_mm'] for v in bearings)
    engagement = min(v['retained_engagement_with_conservative_tilt_mm'] for v in bearings)
    rotation_gap = min(v['certified_continuous_rotation_gap_mm'] for v in rotations)
    holder_gap = min(v['minimum_clearance_lower_bound_mm'] for v in wall['modules'])
    fixed = [record for module in full['modules']
             for record in module['disconnected_fixed_outer_only_records']]
    trims = [record for module in meta
             for record in module.get('localized_print_start_trims', [])]
    trim_volume = sum(record['actual_removed_shell_volume_mm3'] for record in trims)
    full_rows = '\n'.join(
        f'・{module["layer_count"]}層球：全高{module["heights_checked"]:,}高さ、'
        f'内部の未接続部分の検出{len(module["disconnected_internal_records"])}件。'
        for module in full['modules'])
    sizes = []
    procedures = []
    for m in meta:
        n = m['layer_count']
        radii = ' / '.join(f'{2*r:.1f}' for r in m['layer_radii_mm'])
        gaps = ' / '.join(f'{r:.2f}' for r in m['clear_radial_gaps_mm'])
        sizes.append(f'{n}層球：各層の基準径 {radii} mm、3個。球殻同士の公称空隙 {gaps} mm。')
        procedures.append(f'【{n}層球、外側基準径{2*m["outer_nominal_radius_mm"]:.0f} mm】')
        for host in reversed(range(n-1)):
            wires = [w for w in m['release_wires'] if w['host_index'] == host]
            wires.sort(key=lambda w:w['sequence_index'])
            f = next(w for w in wires if w['kind']=='front_window')
            bottom = [w for w in wires if w['kind']=='bottom_starter']
            procedures.append(
                f'  第{host+2}層を解放：{f["sequence_index"]+1:02d} 前面'
                f'（配置角{f["azimuth_degrees"]%360:.1f}度）→ '
                f'{bottom[0]["sequence_index"]+1:02d} 底部左 → '
                f'{bottom[1]["sequence_index"]+1:02d} 底部右。'
                f'底部つまみのY={bottom[0]["grip_point"][1]:.3f} mm。')
    size_text = '\n'.join(sizes)
    procedure_text = '\n'.join(procedures)
    fixed_text = (f'検査した断面には、固定外殻だけに属する未接続部分の記録が{len(fixed)}件あります。'
                  'これは同じ箇所を異なる高さで記録した場合を含み、独立した欠陥の個数ではありません。'
                  '固定外殻と外側装飾には、スライサーで外部サポートを設定してください。'
                  '詳細な高さ・位置・体積はfull_height_validation.jsonに残しています。')
    readme = f'''象牙風多層球のペン立て V5 — 造形途中の内部孤立箇所を修正
2026-10-25 / 単位 mm

「印刷したら多層球が壊れた」という報告を受け、造形順序の追加検査を行った改訂です。
今回印刷された版、破損した場所・工程、プリンター、樹脂はまだ確認できていません。
独立した全高検査で、保存済みV4には前面の仮支持線と内側球殻の一部に、
他の部分につながる前に現れる材料があることを確認しました。V5ではその形を修正しています。
V4前面支持の最初の材料は、最初の恒久殻との接触より最大{old_front_lead:.3f} mm手前の高さから始まっていました。
前回の開始直後だけの検査では、この造形途中の問題を見逃していました。
この発見がユーザーの実物の破損原因を確定した、という意味ではありません。

本ファイルは幾何学検査を通した設計改訂です。使用プリンター・樹脂での実機試験は未実施です。
先に付属の5層球試験片とPRINT_TEST_JA.txtで、印刷後の状態と支持の除去性を確認してください。

1. V4からの変更と維持した寸法

・前面の仮支持線を、下側の窓縁の接点から上へ続けて造形される配置・経路へ変更。
  接点、細い首、中央のつまみの各部分を積層途中の高さで検査して選定しています。
  途中高さでの接続検査と、完成後の切断・取り出し検査を別々に行います。
  接点から曲線胴の接続中心までの長さを0.48から0.62 mmへ延長し、
  胴が破断首へ重ならない形にしています。切断後の片と球殻との初期の重なりに加え、
  正面へ引き出す全移動経路の干渉を実形状で検査しています。
・内側球殻の下向きの微小先端は、下側の既存材料につながる高さまで局所的に平らに修正。
  3種類の球で記録された局所修正は{len(trims)}件、除去体積の合計は{trim_volume:.6f} mm³です。
  数値は3種類各1個の合計であり、9個配置したペン立て全体では3倍になります。
  以前の局所修正を含みます。同じ先端の再調整があるため、独立した欠陥の個数ではありません。
  回転ピン・軸受・穴の保護形状に触れる材料を切除対象から外し、実際の交差を検査しています。
  局所的に平らにした花弁先端の縁について、残った最小肉厚を別途測定したものではありません。
・全高0.05 mm間隔の検査を追加。さらに内部殻と仮支持のメッシュの局所最低点直後を
  0.0001／0.001／0.005／0.025 mmの4点で検査します。
  1e-10 mm³を超える、内部可動層または仮支持を含む未接続部分があれば出力を止めます。
  この幾何学的な検出限界は、プリンターが実際に再現できる寸法を表すものではありません。
・球殻の基本厚は全層1.20 mm、通常の透かし穴同士の内面側の梁幅の目標は1.10 mmを維持。
  V3の基本厚0.64／0.70 mmからの補強を引き継いでいます。
  これらは軸受穴、作業窓、底面の受け座、装飾を含む全箇所の最小厚さ・幅ではありません。
・回転ピン直径1.20 mm、軸受穴直径2.00 mm、軸受外径3.30 mmを維持。
  軸受の公称側壁と盲端は各0.65 mm、洗浄孔直径0.70 mmです。
・作業窓は幅2.40 mm。左右に基本厚1.20 mm・幅1.05 mmの補強帯を備えます。
・各層間の仮支持は前面1本＋底部左右2本。前面は胴径0.55 mm・首径0.40 mm、
  底部は胴0.82×0.76 mm・首0.74×0.54 mm。底部2本が初期の独立した支持経路です。

2. 全体の構成・寸法

外形の実測：{dims} mm（X × Y × Z、装飾を含む）。
造形メッシュ：{geom['vertices']:,}頂点、{geom['triangles']:,}三角形。
{size_text}

基準径には浮彫りと取り付け枠を含みません。公称空隙は球殻の基本面同士の値で、
軸受・支持残り・リブを含む機構全体の最小隙間とは別です。
9個の固定外殻の内側に計27個の可動層があります。各層の運動は左右の共通軸まわりの1自由度です。
本体は内外の立体レースと接続支柱で球の受け枠を支え、壁の奥行きは約12～15 mmです。
本体の糸状装飾の設計幅0.27～0.62 mmと、今回補強した球殻の厚さは別の寸法です。

3. 吊り下げ造形の向きと試験片

本体の上方向は出力座標+Zです。底面を先に、続いて胴部、上縁を造形します。
槽底から照射する装置で造形物が天地逆に吊られて見えても、このモデル内の積層順です。
内部支持の検証は球ローカル+Y方向、すなわち本体+Z方向を前提とします。
大きく傾ける、上縁を先にするなど造形順を変える場合は、内部の最初の出現位置を再確認してください。

試験片：ivory_sphere_v5_test_coupon.3mf / .stl
26 mmの5層球を、32×20×2 mmの台座と固定外殻の受け枠に載せています。
全体の実測寸法は{coupon_dims} mmです。前窓は出力−Y側、造形順は台座から+Z方向です。
追加枠は固定外殻の取り付けリングにだけ接続しています。可動層、切断箇所、支持の取り出し通路を避けています。
試験片は球の肉厚、内部の剥離への耐性、隙間、支持の除去性を確認するためのものです。
ペン立て全体の剥離面積や本体のたわみを再現する試験ではありません。

{fixed_text}
外部サポートを追加するときは、可動層の隙間・軸受内部・正面の作業窓・取り出し通路をふさがないようにします。
各層の最下部は左右2個の受け座から始まり、その0.08 mm後に本来の球殻が始まる構成です。
露光、積層ピッチ、リフト・剥離、洗浄、二次硬化は使用するプリンターと樹脂の条件に従って設定してください。

4. 仮支持を外す順序

小球1個：支持6本、破断首12か所。中球1個：支持9本、首18か所。大球1個：支持12本、首24か所。
9個全体：支持81本、首162か所（前面支持27本、底部支持54本）。
各支持の両端に長さ0.14 mmの細い首があり、その前後で断面が変わります。
この形により切断・曲げを接点付近へ集中させますが、実際の破断位置や荷重を保証するものではありません。
底部の2本はX=±0.56 mmです。底部リブ、窓縁の補強帯、受け座、軸、軸受、取り付け枠は恒久部品です。

最も内側の層間から外側へ進みます。1つの層間では前面1本、底左、底右の順です。
1本ずつ中央のつまみを保持し、両端の首を切り離してから、正面へ取り出します。
球殻全体をねじって支持をまとめて折らず、外側の支持が残っている間は窓の初期向きを保ちます。
両端を切った支持片と未除去の首を数え、各層間の3本を外してから回転を確認します。
切断部の残りは設計値0.15 mm、回転の検証では0.20 mmまでの残り形状を含めています。
大きな残りや欠けがある場合は、その形状に合わせて整える必要があります。

工具経路は先端径0.20 mm・軸径0.50 mmの細い工具を基準に確認しています。
市販のニッパーやペンチの本体、指、あらゆる刃先の作業空間を保証する数値ではありません。
適切な工具を使い、樹脂の指定に合った洗浄・硬化段階で作業してください。

下記は最外層を第1層とする番号です。球を正面から見て右0度・上90度・反時計回りです。
底部Yは球中心を原点とするローカル座標。角度はつまみの位置そのものとは多少異なります。
{procedure_text}

5. 幾何学的な検証結果

・3種類とも支持付きで1つの閉じた固体、除去後はそれぞれ3／4／5個の層。
・9層間の各々で、造形初期の独立した底部支持2経路を確認。
・各可動層の開始位置から0.01／0.04／0.079／0.081／0.10／0.20／0.24 mmの指定断面を検査。
・全高の規則間隔と局所最低点直後を検査した結果：
{full_rows}
・軸受の実形状から得た半径方向の最小隙間下限：{radial_gap:.6f} mm。
・反対側への軸方向移動と傾きを考慮した軸の最小保持量：{engagement:.6f} mm。
・許容した支持残りを含む、中心をそろえた連続360度回転の最小隙間下限：{rotation_gap:.6f} mm。
・本体と可動球の全角度・軸受遊び累積を含む隙間下限：{holder_gap:.6f} mm。
・指定工具の到達、支持片の正面への連続取り出し、本体による通路の妨げを確認。
・最終の出力精度で閉じた面、向き、接続、退化面を確認。
・STL、3MF、GLBを書き出した後に読み直し、全三角形の座標・向き・単位変換を照合。

上記は樹脂の強度試験や有限要素解析ではありません。回転の値は宣言した配置と残り形状についての幾何学的下限です。
収縮・露光による隙間の閉塞、摩擦、未硬化樹脂の強度、剥離荷重、実際の破断面は検証対象外です。
全高検査は指定した高さの有限個の解析です。全ての連続高さ、任意のスライス位相、
全画素、造形の張り出し角度、装置固有の支持能力を証明する検査ではありません。
固定外殻単独の孤立記録は内部の合否と分け、外部サポートを必要とする箇所として残しています。
球殻の強度改善は、通常の透かし間の幅、球殻の基本厚、実断面、接続形状などを比較したものです。
機種、樹脂、積層ピッチ、露光、失敗した段階が不明なため、条件に固有の成功率や安全率は示していません。

6. データ形式

造形用本体：{STEM}.3mf / .stl。支持81本を含みます。
3MFは単位mmを明示。STLはmm・倍率100%で読み込んでください。
GLBは表示用でm単位・Y上方向です。造形用には3MFまたはSTLを使用します。
説明画像の色分けと断面は構造を見せるための表示です。印刷モデルの欠損や多材料指定を意味しません。
STL・GLBは各々の通常ZIPを展開してください。プロジェクトZIPには再生成ソース、監査報告、球の部品キャッシュ、V3比較資料が入っています。
保存先はGoogle Driveのworkspace/penholder_2/V5です。

7. 再生成・報告書

このREADMEがあるフォルダーで実行：
  python -m pip install -r source/requirements.txt
  python source/rebuild.py
画像も再生成：
  python source/rebuild.py --render
配布ファイルを生成：
  python source/package_delivery.py
試験片だけ再生成：
  python source/build_test_coupon.py

Pythonライブラリーの検証時バージョンはsource/requirements.txtに記載。
Linuxのヘッドレス描画ではXvfb等の描画環境が必要です。
主要報告：motion_validation、early_support_validation、full_height_validation、strength_validation、geometry_validation、
holder_motion_clearance、holder_service_access、precision_repairs_scope、test_coupon_validation。
すべてreports内のJSONです。ファイルのSHA256と検査対象・制約を一緒に参照してください。
球の各部品と作業番号の原本はcache/spheres/sphere_3layers.json、sphere_4layers.json、sphere_5layers.jsonです。
JSONのwire_indexとsequence_indexは0から始まり、この説明書の作業番号は1から始まります。
reference/v3には比較に必要な前版の資料を保存し、最終V5の部品と分けています。
full_height_v4_baseline.jsonは今回追加した検査をV4へ適用した比較記録です。
この比較記録のpassed=falseは既知の問題を正しく検出した結果で、V5の合否とは別です。

8. 造形上の参考資料

Formlabs, Form 4 Design Guide:
https://formlabs.com/white-papers/form-4-design-guide/
Formlabs, How supports work in SLA printing:
https://formlabs.com/support/How-supports-work-in-SLA-printing/
これらの一般的な指針は、使用条件の異なる本モデルの造形成功を保証する資料ではありません。
V5の寸法と上記の検査数値は、付属のソースと実メッシュの検証記録によります。
'''
    trial = f'''V5 5層球 — 最初の試験印刷
2026-10-25

使うファイル：ivory_sphere_v5_test_coupon.3mf（推奨）または同名.stl。
実測外形：{coupon_dims} mm。台座32×20×2 mm、球の基本外径26 mm、5層。
球殻の基本厚は1.20 mm。仮支持は12本、破断首24か所です。

1. mm単位、倍率100%で読み込みます。台座を下、球を上、モデル+Zへ積層します。
   SLA装置内で逆さに吊られる見え方でも、台座から先に印刷する順序です。
2. 固定外殻と外側装飾の最下点・張り出しに外部サポートを設定し、全スライスを確認します。
   内部の層間、軸受穴、正面の作業窓、支持を引き出す通路へ外部サポートを追加しないでください。
3. プリンターと樹脂で実績のある露光・積層・剥離・洗浄・二次硬化条件を使います。
   本データに特定機種の露光秒数やリフト条件は設定していません。
4. 印刷後は球殻の欠け、反り、層同士の融着、底部支持の切断を写真と一緒に確認します。
   印刷中・台座取り外し・洗浄・支持除去・二次硬化のどの段階で変化したか記録してください。
5. 樹脂の指定に適した状態で、最内側の層間から、前面1本→底左→底右を順に外します。
   それぞれ中央のつまみを保持し、両端の首で切り離してから正面へ取り出します。
   球殻をねじってまとめて折らず、3本外した時点でその層の回転を確認します。
6. 4組の層間、計12本／24首を終え、内側4層が左右の軸に保持されたまま回るか確認します。
   残り突起が大きい、層が融着している、割れがある場合は無理に回さず、その位置を記録します。

工具通路の基準は先端径0.20 mm、軸径0.50 mmです。一般的なニッパー本体の通路を保証しません。
台座・受け枠・軸受・窓脇の補強帯は完成後も残す部品です。
支持番号と首の場所はREADME_JA.txt、機構図、sphere_5layers.jsonを参照してください。

試験片で確認したい項目：
  [ ] 印刷後の球殻に割れ・崩れがない
  [ ] 洗浄・指定の硬化処理で薄い縁が欠けない
  [ ] 支持12本を球殻を割らずに取り外せる
  [ ] 内側4層が回転し、軸から脱落しない
  [ ] 窓と軸受の隙間を洗浄できる

記録欄：
  プリンター：
  樹脂名・色：
  積層ピッチ：
  通常露光／底部露光：
  リフト・剥離条件：
  洗浄／二次硬化：
  破綻した段階・位置：
  写真：

この試験片は、本体全体の剥離面積や壁剛性を再現しません。
全体の印刷は、試験片の結果を踏まえて外部サポートと条件を確認してから進めてください。
V5は造形途中の接続も含め、幾何学的に補強・検証した設計です。実機試験に合格したという意味ではありません。
'''
    (ROOT/'README_JA.txt').write_text(readme, encoding='utf-8')
    (ROOT/'PRINT_TEST_JA.txt').write_text(trial, encoding='utf-8')
    print('Wrote README_JA.txt and PRINT_TEST_JA.txt from validated V5 metadata.')


if __name__ == '__main__':
    main()
