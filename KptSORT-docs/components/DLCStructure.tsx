import React from 'react'
import { FileTree } from 'nextra/components'

// スタイル定義
const highlightStyle = { color: "#991b1b", textDecoration: "none", fontWeight: "bold" };
const linkStyleBase = { textDecoration: "underline", cursor: "pointer", display: "inline-flex", alignItems: "baseline", gap: "4px" };
const linkDefaultColor = { color: "#0070f3" };

export const DLCStructure = ({ highlights = [] }: { highlights?: string[] }) => {

    const f = (name: string, url?: string) => {
        const isHighlighted = highlights.includes(name);

        if (url) {
            const finalLinkStyle = {
                ...linkStyleBase,
                ...(isHighlighted ? highlightStyle : linkDefaultColor)
            };
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" style={finalLinkStyle}>
                    {name}
                    <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" height="0.8em" style={{ display: 'inline', verticalAlign: 'baseline' }}>
                        <path d="M7 17L17 7"></path><path d="M7 7h10v10"></path>
                    </svg>
                </a>
            );
        }

        return isHighlighted ? <span style={highlightStyle}>{name}</span> : name;
    };

    const hasH = (names: string[]) => names.some(name => highlights.includes(name));

    // --- ファイルリスト定義の最適化 ---

    // 1. 各設定ファイル
    const configFiles = ["config.yaml", "pytorch_config.yaml", "inference_cfg.yaml"];

    // 2. モデル関連 (shuffle1 / shuffle2)
    const s1Files = ["pytorch_config.yaml", "inference_cfg.yaml", "LabeledImages_DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-400"];
    const s2Files = ["pytorch_config.yaml", "inference_cfg.yaml", "LabeledImages_DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-50"];
    const iteration0Files = [...s1Files, ...s2Files, "UnaugmentedDataSet_0902_BUCTDSep2"];

    // 3. 動画解析結果 (BU / CTD)
    const buFiles = [
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_assemblies.pickle",
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_el.csv",
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_el.h5",
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_el.pickle",
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_full.pickle",
        "1110PBS_29_1DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-750_meta.pickle"
    ];

    const ctdFiles = [
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_assemblies.pickle",
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_el.csv",
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_el.h5",
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_el.pickle",
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_full.pickle",
        "1110PBS_29_1DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-20_meta.pickle"
    ];

    const pbsAll = ["1110PBS_29_1.mp4", ...buFiles, ...ctdFiles];

    return (
        <FileTree>
            {/* dlc-models-pytorch */}
            <FileTree.Folder name="dlc-models-pytorch" defaultOpen={hasH(iteration0Files)}>
                <FileTree.Folder name="iteration-0" defaultOpen={hasH(iteration0Files)}>
                    <FileTree.Folder
                        name="0902_BUCTDSep2-trainset70shuffle1"
                        defaultOpen={hasH(s1Files)}
                    >
                        <FileTree.Folder name={f("test")} defaultOpen={hasH(["inference_cfg.yaml"])}>
                            <FileTree.File name={f("inference_cfg.yaml")} />
                        </FileTree.Folder>
                        <FileTree.Folder name={f("train")} defaultOpen={hasH(["pytorch_config.yaml"])}>
                            <FileTree.File name={f("pytorch_config.yaml")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                    <FileTree.Folder
                        name="0902_BUCTDSep2-trainset70shuffle2"
                        defaultOpen={hasH(s2Files)}
                    >
                        <FileTree.Folder name={f("test")} defaultOpen={hasH(["inference_cfg.yaml"])}>
                            <FileTree.File name={f("inference_cfg.yaml")} />
                        </FileTree.Folder>
                        <FileTree.Folder name={f("train")} defaultOpen={hasH(["pytorch_config.yaml"])}>
                            <FileTree.File name={f("pytorch_config.yaml")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                </FileTree.Folder>
            </FileTree.Folder>

            {/* evaluation-results-pytorch */}
            <FileTree.Folder name="evaluation-results-pytorch" defaultOpen={hasH(iteration0Files)}>
                <FileTree.Folder name="iteration-0" defaultOpen={hasH(iteration0Files)}>
                    <FileTree.Folder
                        name="0902_BUCTDSep2-trainset70shuffle1"
                        defaultOpen={hasH(["LabeledImages_DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-400"])}
                    >
                        <FileTree.Folder name={f("LabeledImages_DLC_DekrW48_0902_BUCTDSep2shuffle1_snapshot_best-400")}>{""}</FileTree.Folder>
                    </FileTree.Folder>
                    <FileTree.Folder
                        name="0902_BUCTDSep2-trainset70shuffle2"
                        defaultOpen={hasH(["LabeledImages_DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-50"])}
                    >
                        <FileTree.Folder name={f("LabeledImages_DLC_CtdCoamW32_0902_BUCTDSep2shuffle2_snapshot_best-50")}>{""}</FileTree.Folder>
                    </FileTree.Folder>
                </FileTree.Folder>
            </FileTree.Folder>

            {/* labeled-data */}
            <FileTree.Folder name="labeled-data" defaultOpen={hasH(["resized_color_20250723-111000_PBS_19_1h", "resized_color_20250723-111000_PBS_19_1h_labeled"])}>
                <FileTree.Folder name={f("resized_color_20250723-111000_PBS_19_1h")}>{""}</FileTree.Folder>
                <FileTree.Folder name={f("resized_color_20250723-111000_PBS_19_1h_labeled")}>{""}</FileTree.Folder>
            </FileTree.Folder>

            {/* training-datasets */}
            <FileTree.Folder name="training-datasets" defaultOpen={hasH(iteration0Files)}>
                <FileTree.Folder name="iteration-0" defaultOpen={hasH(["UnaugmentedDataSet_0902_BUCTDSep2"])}>
                    <FileTree.Folder name={f("UnaugmentedDataSet_0902_BUCTDSep2")}>{""}</FileTree.Folder>
                </FileTree.Folder>
            </FileTree.Folder>

            {/* videos */}
            <FileTree.Folder name="videos" defaultOpen={hasH(pbsAll)}>
                <FileTree.Folder name="1110PBS_29_1" defaultOpen={hasH(pbsAll)}>
                    <FileTree.File name={f("1110PBS_29_1.mp4")} />

                    <FileTree.Folder name="BU" defaultOpen={hasH(buFiles)}>
                        {buFiles.map(file => <FileTree.File key={file} name={f(file)} />)}
                    </FileTree.Folder>

                    <FileTree.Folder name="CTD" defaultOpen={hasH(ctdFiles)}>
                        {ctdFiles.map(file => <FileTree.File key={file} name={f(file)} />)}
                    </FileTree.Folder>
                </FileTree.Folder>
            </FileTree.Folder>

            <FileTree.File name={f("config.yaml")} />
        </FileTree>
    );
};