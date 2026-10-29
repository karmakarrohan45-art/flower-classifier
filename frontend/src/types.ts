export interface FlowerMetadata {
  name: string;
  scientific_name: string;
  "height (cm)": string;
  "longevity (years)": string;
  "place of origin": string;
  color: string;
  climate: string;
  "cut flowers": string;
  perfumes: string;
  medicine: string;
  "average number of petals": string;
}

export interface Prediction {
  topClass: string;
  confidence: number;
  allClasses: { className: string; confidence: number }[];
  metadata?: FlowerMetadata;
}
